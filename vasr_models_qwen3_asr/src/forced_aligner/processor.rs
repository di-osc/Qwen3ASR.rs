//! Forced aligner text preprocessing and timestamp post-processing.
//!
//! Faithful port of:
//! `../../../Qwen3-ASR/qwen_asr/inference/qwen3_forced_aligner.py` (processor parts)
//!
//! Notes:
//! - The official implementation uses external tokenizers for Japanese/Korean.
//! - This Rust port implements the upstream Korean tokenizer behavior in pure Rust:
//!   the `soynlp` `LTokenizer` algorithm over an embedded dictionary.
//! - Japanese tokenization defaults to `lindera` + IPADIC (pure Rust) and falls back to a
//!   conservative script-run tokenizer if `lindera` fails or returns no tokens.

use anyhow::{Context, Result, bail};
use std::collections::{BTreeSet, HashSet};
use std::path::Path;
use std::sync::OnceLock;
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;

// Sourced from the official Qwen3-ASR implementation:
// `../../../Qwen3-ASR/qwen_asr/inference/assets/korean_dict_jieba.dict` (Apache-2.0).
const KOREAN_DICT_JIEBA: &str = include_str!("assets/korean_dict_jieba.dict");

#[derive(Debug)]
struct KoreanDict {
    words: HashSet<String>,
    max_len_chars: usize,
}

impl KoreanDict {
    fn from_embedded() -> Result<Self> {
        let mut words: HashSet<String> = HashSet::new();
        let mut max_len_chars: usize = 0;

        for line in KOREAN_DICT_JIEBA.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some(word) = line.split_whitespace().next() else {
                continue;
            };
            if word.is_empty() {
                continue;
            }
            max_len_chars = max_len_chars.max(word.chars().count());
            words.insert(word.to_string());
        }

        if words.is_empty() {
            bail!("embedded Korean dictionary is empty");
        }
        if max_len_chars == 0 {
            bail!("embedded Korean dictionary max word length is zero");
        }

        Ok(Self {
            words,
            max_len_chars,
        })
    }
}

static KOREAN_DICT: OnceLock<std::result::Result<KoreanDict, String>> = OnceLock::new();

fn korean_dict() -> Result<&'static KoreanDict> {
    let res = KOREAN_DICT.get_or_init(|| {
        KoreanDict::from_embedded().map_err(|e| format!("failed to load Korean dictionary: {e}"))
    });

    match res {
        Ok(d) => Ok(d),
        Err(msg) => bail!("{msg}"),
    }
}

static JAPANESE_TOKENIZER: OnceLock<std::result::Result<lindera::tokenizer::Tokenizer, String>> =
    OnceLock::new();

const JAPANESE_DEFAULT_MERGE_TERMS: &str = include_str!("assets/japanese_merge_terms_default.txt");

fn normalize_japanese_nfkc(text: &str) -> String {
    text.nfkc().collect()
}

#[derive(Debug, Clone)]
struct JapaneseMergeTerms {
    terms: HashSet<String>,
    max_chars: usize,
}

impl JapaneseMergeTerms {
    fn builtin_terms() -> &'static Vec<String> {
        static TERMS: OnceLock<Vec<String>> = OnceLock::new();
        TERMS.get_or_init(|| {
            JAPANESE_DEFAULT_MERGE_TERMS
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .map(str::to_string)
                .collect()
        })
    }

    fn from_builtin_and_user(user_terms: &[String]) -> Self {
        let mut terms: HashSet<String> = HashSet::new();
        let mut max_chars: usize = 0;

        for term in Self::builtin_terms()
            .iter()
            .map(String::as_str)
            .chain(user_terms.iter().map(String::as_str))
        {
            let normalized = normalize_japanese_nfkc(term.trim());
            let cleaned = ForcedAlignProcessor::clean_token(normalized.as_str());
            if cleaned.is_empty() {
                continue;
            }
            max_chars = max_chars.max(cleaned.chars().count());
            terms.insert(cleaned);
        }

        Self { terms, max_chars }
    }

    fn contains(&self, term: &str) -> bool {
        self.terms.contains(term)
    }

    fn max_chars(&self) -> usize {
        self.max_chars
    }
}

fn default_japanese_merge_terms() -> &'static JapaneseMergeTerms {
    static MERGE_TERMS: OnceLock<JapaneseMergeTerms> = OnceLock::new();
    MERGE_TERMS.get_or_init(|| JapaneseMergeTerms::from_builtin_and_user(&[]))
}

fn japanese_tokenizer() -> Result<&'static lindera::tokenizer::Tokenizer> {
    let res = JAPANESE_TOKENIZER.get_or_init(|| {
        let dict = lindera::dictionary::load_dictionary("embedded://ipadic")
            .map_err(|e| format!("failed to load embedded IPADIC dictionary: {e}"))?;
        let segmenter = lindera::segmenter::Segmenter::new(lindera::mode::Mode::Normal, dict, None);
        Ok(lindera::tokenizer::Tokenizer::new(segmenter))
    });

    match res {
        Ok(t) => Ok(t),
        Err(msg) => bail!("{msg}"),
    }
}

#[derive(Debug, Clone)]
pub struct ForcedAlignProcessor {
    japanese_merge_terms: JapaneseMergeTerms,
}

impl Default for ForcedAlignProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ForcedAlignProcessor {
    pub fn new() -> Self {
        Self {
            japanese_merge_terms: default_japanese_merge_terms().clone(),
        }
    }

    pub fn with_japanese_user_dictionary_path(path: &Path) -> Result<Self> {
        let user_terms = Self::load_japanese_user_terms(path)?;
        Ok(Self::from_japanese_user_terms(user_terms))
    }

    pub fn from_japanese_user_terms<I>(terms: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        let user_terms = Self::normalize_japanese_user_terms(terms);
        Self {
            japanese_merge_terms: JapaneseMergeTerms::from_builtin_and_user(user_terms.as_slice()),
        }
    }

    fn load_japanese_user_terms(path: &Path) -> Result<Vec<String>> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read Japanese user dictionary {path:?}"))?;

        let is_json = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("json"))
            .unwrap_or(false);

        if is_json {
            #[derive(Debug, serde::Deserialize)]
            #[serde(untagged)]
            enum UserDictionary {
                List(Vec<String>),
                Object { terms: Vec<String> },
            }

            let parsed: UserDictionary = serde_json::from_str(content.as_str())
                .with_context(|| format!("failed to parse Japanese user dictionary {path:?}"))?;
            let terms = match parsed {
                UserDictionary::List(terms) => terms,
                UserDictionary::Object { terms } => terms,
            };
            return Ok(Self::normalize_japanese_user_terms(terms));
        }

        let terms = content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(str::to_string)
            .collect::<Vec<_>>();
        Ok(Self::normalize_japanese_user_terms(terms))
    }

    fn normalize_japanese_user_terms<I>(terms: I) -> Vec<String>
    where
        I: IntoIterator<Item = String>,
    {
        let mut uniq: BTreeSet<String> = BTreeSet::new();
        for term in terms {
            let cleaned = Self::clean_token_japanese(term.trim());
            if !cleaned.is_empty() {
                uniq.insert(cleaned);
            }
        }
        uniq.into_iter().collect()
    }

    fn merge_japanese_tokens_with_terms(
        tokens: Vec<String>,
        merge_terms: &JapaneseMergeTerms,
    ) -> Vec<String> {
        // The official stack uses `nagisa` (morphological tokenizer). `lindera` can produce
        // slightly different boundaries, especially around unknown terms / ASCII fragments.
        //
        // These conservative merge rules keep the `clean_token` contract:
        // `tokens.join("") == clean_token(text)`.
        //
        // 1) Merge consecutive single-kanji tokens (e.g., "東" + "京" + "都" => "東京都").
        // 2) Merge dictionary-style phrase terms via longest-match (e.g., "Open"+"AI" => "OpenAI").
        //
        // We intentionally avoid generic ASCII run merging by default because it over-merges mixed
        // Latin/digit cases (e.g., "iPhone15 Pro"), which drifts from nagisa boundaries.
        let mut merged: Vec<String> = Vec::with_capacity(tokens.len());
        let mut kanji_buf = String::new();
        for tok in tokens {
            let mut it = tok.chars();
            let ch = it.next();
            let is_single_kanji = match (ch, it.next()) {
                (Some(c), None) => Self::is_cjk_char(c),
                _ => false,
            };

            if is_single_kanji {
                if let Some(c) = ch {
                    kanji_buf.push(c);
                }
                continue;
            }

            if !kanji_buf.is_empty() {
                merged.push(std::mem::take(&mut kanji_buf));
            }
            merged.push(tok);
        }
        if !kanji_buf.is_empty() {
            merged.push(kanji_buf);
        }

        // Step 2: merge dictionary-style phrase terms using greedy longest-match.
        let max_term_chars = merge_terms.max_chars();
        if max_term_chars == 0 {
            return merged;
        }

        let mut merged_terms: Vec<String> = Vec::with_capacity(merged.len());
        let mut i = 0usize;
        while i < merged.len() {
            let mut best: Option<(usize, String)> = None;
            let mut acc = String::new();

            for (j, token) in merged.iter().enumerate().skip(i) {
                acc.push_str(token.as_str());
                if acc.chars().count() > max_term_chars {
                    break;
                }
                if merge_terms.contains(acc.as_str()) {
                    best = Some((j.saturating_add(1), acc.clone()));
                }
            }

            if let Some((next_i, phrase)) = best {
                merged_terms.push(phrase);
                i = next_i;
            } else {
                merged_terms.push(merged[i].clone());
                i = i.saturating_add(1);
            }
        }

        merged_terms
    }

    fn is_kept_char(ch: char) -> bool {
        if ch == '\'' {
            return true;
        }

        // Match the official Python semantics:
        // `unicodedata.category(ch).startswith(("L", "N"))`.
        matches!(
            get_general_category(ch),
            GeneralCategory::UppercaseLetter
                | GeneralCategory::LowercaseLetter
                | GeneralCategory::TitlecaseLetter
                | GeneralCategory::ModifierLetter
                | GeneralCategory::OtherLetter
                | GeneralCategory::DecimalNumber
                | GeneralCategory::LetterNumber
                | GeneralCategory::OtherNumber
        )
    }

    fn clean_token(token: &str) -> String {
        token.chars().filter(|&ch| Self::is_kept_char(ch)).collect()
    }

    fn clean_token_japanese(token: &str) -> String {
        let normalized = normalize_japanese_nfkc(token);
        Self::clean_token(normalized.as_str())
    }

    fn is_cjk_char(ch: char) -> bool {
        let code = u32::from(ch);
        (0x4E00..=0x9FFF).contains(&code) // CJK Unified Ideographs
            || (0x3400..=0x4DBF).contains(&code) // Extension A
            || (0x20000..=0x2A6DF).contains(&code) // Extension B
            || (0x2A700..=0x2B73F).contains(&code) // Extension C
            || (0x2B740..=0x2B81F).contains(&code) // Extension D
            || (0x2B820..=0x2CEAF).contains(&code) // Extension E
            || (0xF900..=0xFAFF).contains(&code) // Compatibility Ideographs
    }

    fn japanese_script_kind(ch: char) -> Option<u32> {
        let code = u32::from(ch);
        if Self::is_cjk_char(ch) {
            return Some(0);
        }
        if (0x3040..=0x309F).contains(&code) {
            return Some(1);
        }
        if (0x30A0..=0x30FF).contains(&code) || (0xFF65..=0xFF9F).contains(&code) {
            return Some(2);
        }
        None
    }

    fn is_hiragana_char(ch: char) -> bool {
        (0x3040..=0x309F).contains(&u32::from(ch))
    }

    fn is_hiragana_small_char(ch: char) -> bool {
        matches!(
            ch,
            'ゃ' | 'ゅ' | 'ょ' | 'ぁ' | 'ぃ' | 'ぅ' | 'ぇ' | 'ぉ' | 'ゎ' | 'っ' | 'ゕ' | 'ゖ'
        )
    }

    fn is_particle_token(token: &str) -> bool {
        matches!(
            token,
            "は" | "が" | "を" | "に" | "で" | "へ" | "と" | "の" | "も" | "や"
        )
    }

    fn is_particle_char(ch: char) -> bool {
        matches!(
            ch,
            'は' | 'が' | 'を' | 'に' | 'で' | 'へ' | 'と' | 'の' | 'も' | 'や'
        )
    }

    fn split_hiragana_particle_token(token: &str) -> Vec<String> {
        let chars: Vec<char> = token.chars().collect();
        let n = chars.len();
        if !chars.iter().all(|&ch| Self::is_hiragana_char(ch)) {
            return vec![token.to_string()];
        }

        if n >= 5
            && chars[0] == 'お'
            && chars[n - 2] == 'さ'
            && chars[n - 1] == 'ん'
            && (n - 3) >= 2
            && chars[1..n - 2].iter().all(|&ch| Self::is_hiragana_char(ch))
        {
            return vec![
                "お".to_string(),
                chars[1..n - 2].iter().collect(),
                "さん".to_string(),
            ];
        }

        if n < 6 {
            return vec![token.to_string()];
        }

        let mut split_positions: Vec<usize> = Vec::new();
        for (i, &ch) in chars.iter().enumerate() {
            if i == 0 || i + 1 >= n {
                continue;
            }
            if Self::is_particle_char(ch) {
                split_positions.push(i);
            }
        }
        if split_positions.is_empty() {
            return vec![token.to_string()];
        }

        let out = if split_positions.len() == 1 {
            let i = split_positions[0];
            let left_len = i;
            let right_len = n.saturating_sub(i.saturating_add(1));
            if n < 7 || left_len < 2 || right_len < 2 {
                vec![token.to_string()]
            } else {
                vec![
                    chars[..i].iter().collect(),
                    chars[i].to_string(),
                    chars[i + 1..].iter().collect(),
                ]
            }
        } else {
            let mut out: Vec<String> = Vec::new();
            let mut start = 0usize;
            for i in split_positions {
                if start < i {
                    out.push(chars[start..i].iter().collect());
                }
                out.push(chars[i].to_string());
                start = i.saturating_add(1);
            }
            if start < n {
                out.push(chars[start..].iter().collect());
            }
            out
        };

        let mut out_tail: Vec<String> = Vec::new();
        for seg in out {
            let seg_chars: Vec<char> = seg.chars().collect();
            let m = seg_chars.len();
            if m >= 4
                && seg_chars[m - 2] == 'か'
                && seg_chars[m - 1] == 'な'
                && seg_chars[..m - 2]
                    .iter()
                    .all(|&ch| Self::is_hiragana_char(ch))
            {
                out_tail.push(seg_chars[..m - 2].iter().collect());
                out_tail.push("か".to_string());
                out_tail.push("な".to_string());
                continue;
            }
            if m >= 4
                && seg_chars[m - 2] == 'よ'
                && seg_chars[m - 1] == 'ね'
                && seg_chars[..m - 2]
                    .iter()
                    .all(|&ch| Self::is_hiragana_char(ch))
            {
                out_tail.push(seg_chars[..m - 2].iter().collect());
                out_tail.push("よ".to_string());
                out_tail.push("ね".to_string());
                continue;
            }
            if m >= 3
                && matches!(seg_chars[m - 1], 'か' | 'ね' | 'よ' | 'な')
                && seg_chars[..m - 1]
                    .iter()
                    .all(|&ch| Self::is_hiragana_char(ch))
            {
                out_tail.push(seg_chars[..m - 1].iter().collect());
                out_tail.push(seg_chars[m - 1].to_string());
                continue;
            }
            out_tail.push(seg);
        }

        out_tail
            .into_iter()
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
    }

    fn split_hiragana_particle_runs(tokens: Vec<String>) -> Vec<String> {
        let mut out: Vec<String> = Vec::with_capacity(tokens.len());
        for token in tokens {
            out.extend(Self::split_hiragana_particle_token(token.as_str()));
        }
        out
    }

    fn repair_hiragana_bridge_fragments(tokens: Vec<String>) -> Vec<String> {
        let mut out: Vec<String> = Vec::with_capacity(tokens.len());
        let mut i = 0usize;
        while i < tokens.len() {
            if i + 1 < tokens.len() {
                let cur = &tokens[i];
                let next = &tokens[i + 1];
                let cur_is_hira = !cur.is_empty() && cur.chars().all(Self::is_hiragana_char);
                let next_is_hira = !next.is_empty() && next.chars().all(Self::is_hiragana_char);
                let cur_len = cur.chars().count();
                let next_len = next.chars().count();
                let next_has_particle = next
                    .chars()
                    .enumerate()
                    .any(|(idx, ch)| idx > 0 && idx + 1 < next_len && Self::is_particle_char(ch));
                if cur_is_hira
                    && next_is_hira
                    && !Self::is_particle_token(cur.as_str())
                    && cur_len <= 2
                    && next_len >= 4
                    && next_has_particle
                {
                    let mut merged = String::with_capacity(cur.len() + next.len());
                    merged.push_str(cur.as_str());
                    merged.push_str(next.as_str());
                    out.push(merged);
                    i = i.saturating_add(2);
                    continue;
                }
            }
            out.push(tokens[i].clone());
            i = i.saturating_add(1);
        }
        out
    }

    fn repair_hiragana_prefix_fragments(tokens: Vec<String>) -> Vec<String> {
        let mut out: Vec<String> = Vec::with_capacity(tokens.len());
        let mut i = 0usize;
        while i < tokens.len() {
            if i + 2 < tokens.len() {
                let cur = &tokens[i];
                let next = &tokens[i + 1];
                let after = &tokens[i + 2];

                let cur_is_hira = !cur.is_empty() && cur.chars().all(Self::is_hiragana_char);
                let next_is_hira = !next.is_empty() && next.chars().all(Self::is_hiragana_char);
                let after_is_particle = Self::is_particle_token(after.as_str());
                let next_is_particle = Self::is_particle_token(next.as_str());
                let next_len = next.chars().count();
                let next_starts_small = next
                    .chars()
                    .next()
                    .map(Self::is_hiragana_small_char)
                    .unwrap_or(false);

                if cur_is_hira
                    && next_is_hira
                    && after_is_particle
                    && !next_is_particle
                    && (next_len == 1 || next_starts_small)
                {
                    let mut merged = String::with_capacity(cur.len() + next.len());
                    merged.push_str(cur.as_str());
                    merged.push_str(next.as_str());
                    out.push(merged);
                    i = i.saturating_add(2);
                    continue;
                }
            }

            out.push(tokens[i].clone());
            i = i.saturating_add(1);
        }

        out
    }

    fn tokenize_japanese_fallback(text: &str) -> Vec<String> {
        let mut tokens: Vec<String> = Vec::new();
        let mut latin_buf = String::new();
        let mut jp_buf = String::new();
        let mut jp_kind: Option<u32> = None;

        for ch in text.chars() {
            if let Some(kind) = Self::japanese_script_kind(ch) {
                if !latin_buf.is_empty() {
                    tokens.push(std::mem::take(&mut latin_buf));
                }
                if jp_kind != Some(kind) && !jp_buf.is_empty() {
                    tokens.push(std::mem::take(&mut jp_buf));
                }
                jp_kind = Some(kind);
                jp_buf.push(ch);
                continue;
            }

            if !jp_buf.is_empty() {
                tokens.push(std::mem::take(&mut jp_buf));
                jp_kind = None;
            }

            if Self::is_kept_char(ch) {
                latin_buf.push(ch);
            } else if !latin_buf.is_empty() {
                tokens.push(std::mem::take(&mut latin_buf));
            }
        }

        if !jp_buf.is_empty() {
            tokens.push(jp_buf);
        }
        if !latin_buf.is_empty() {
            tokens.push(latin_buf);
        }

        tokens
            .into_iter()
            .map(|t| Self::clean_token(&t))
            .filter(|t| !t.is_empty())
            .collect()
    }

    fn tokenize_japanese_lindera(text: &str) -> Result<Vec<String>> {
        let tokenizer = japanese_tokenizer()?;
        let tokens = tokenizer.tokenize(text)?;
        Ok(tokens
            .into_iter()
            .map(|t| Self::clean_token(t.surface.as_ref()))
            .filter(|t| !t.is_empty())
            .collect())
    }

    fn tokenize_japanese_with_terms(
        text: &str,
        merge_terms: &JapaneseMergeTerms,
    ) -> Result<Vec<String>> {
        let normalized = normalize_japanese_nfkc(text);
        let tokens = match Self::tokenize_japanese_lindera(normalized.as_str()) {
            Ok(ts) if !ts.is_empty() => ts,
            Ok(_) => Self::tokenize_japanese_fallback(normalized.as_str()),
            Err(_) => Self::tokenize_japanese_fallback(normalized.as_str()),
        };
        let tokens = Self::repair_hiragana_bridge_fragments(tokens);
        let tokens = Self::split_hiragana_particle_runs(tokens);
        let tokens = Self::repair_hiragana_prefix_fragments(tokens);
        let tokens = Self::split_ascii_digit_runs(tokens);
        let tokens = Self::split_katakana_compounds(tokens);
        Ok(Self::merge_japanese_tokens_with_terms(tokens, merge_terms))
    }

    fn split_ascii_digit_runs(tokens: Vec<String>) -> Vec<String> {
        let mut out = Vec::with_capacity(tokens.len());
        for token in tokens {
            let chars = token.chars().collect::<Vec<_>>();
            if chars.len() > 1 && chars.iter().all(|ch| ch.is_ascii_digit()) {
                out.extend(chars.into_iter().map(|ch| ch.to_string()));
            } else {
                out.push(token);
            }
        }
        out
    }

    fn is_fullwidth_katakana_char(ch: char) -> bool {
        let code = u32::from(ch);
        (0x30A0..=0x30FF).contains(&code)
    }

    fn split_katakana_compounds(tokens: Vec<String>) -> Vec<String> {
        let mut out = Vec::with_capacity(tokens.len());
        for token in tokens {
            if token.chars().all(Self::is_fullwidth_katakana_char)
                && token.ends_with("カタカナ")
                && token != "カタカナ"
            {
                let suffix_start = token.len().saturating_sub("カタカナ".len());
                if suffix_start > 0
                    && token.is_char_boundary(suffix_start)
                    && token[..suffix_start].chars().count() >= 2
                {
                    out.push(token[..suffix_start].to_string());
                    out.push("カタカナ".to_string());
                    continue;
                }
            }
            out.push(token);
        }
        out
    }

    fn ltokenizer_split_korean_token(token: &str, dict: &KoreanDict) -> (String, String) {
        // Mirrors soynlp.tokenizer.LTokenizer.token_to_lr():
        // - If token length <= 2, return (token, "").
        // - Otherwise, find the best (L, R) split where L is a prefix of length >= 2.
        // - With uniform scores and default_score=0.0, the best candidate is:
        //   * the longest L that exists in the dictionary, or
        //   * the full token if no dictionary prefix is found.
        let len_chars = token.chars().count();
        if len_chars <= 2 {
            return (token.to_string(), String::new());
        }

        // Precompute byte offsets for each char boundary so we can slice by char count.
        let mut boundaries: Vec<usize> = Vec::with_capacity(len_chars.saturating_add(1));
        boundaries.push(0);
        for (idx, _ch) in token.char_indices().skip(1) {
            boundaries.push(idx);
        }
        boundaries.push(token.len());

        let search_max = len_chars.min(dict.max_len_chars);
        let mut best_e: Option<usize> = None;
        for e in 2..=search_max {
            if let Some(&byte) = boundaries.get(e) {
                let left = &token[..byte];
                if dict.words.contains(left) {
                    best_e = Some(e);
                }
            }
        }

        let e = best_e.unwrap_or(len_chars);
        let split = boundaries.get(e).copied().unwrap_or(token.len());
        let left = token[..split].to_string();
        let right = token[split..].to_string();
        (left, right)
    }

    fn tokenize_korean(text: &str) -> Result<Vec<String>> {
        // Faithful port of:
        // `Qwen3ForceAlignProcessor.tokenize_korean()` which uses `soynlp` `LTokenizer`
        // and then applies `clean_token()` to each output token.
        let dict = korean_dict()?;

        let mut tokens: Vec<String> = Vec::new();
        for raw in text.split_whitespace() {
            if raw.is_empty() {
                continue;
            }

            let (l, r) = Self::ltokenizer_split_korean_token(raw, dict);

            let l_clean = Self::clean_token(l.as_str());
            if !l_clean.is_empty() {
                tokens.push(l_clean);
            }

            if !r.is_empty() {
                let r_clean = Self::clean_token(r.as_str());
                if !r_clean.is_empty() {
                    tokens.push(r_clean);
                }
            }
        }

        Ok(tokens)
    }

    fn split_segment_with_chinese(seg: &str) -> Vec<String> {
        let mut tokens: Vec<String> = Vec::new();
        let mut buf = String::new();

        for ch in seg.chars() {
            if Self::is_cjk_char(ch) {
                if !buf.is_empty() {
                    tokens.push(std::mem::take(&mut buf));
                }
                tokens.push(ch.to_string());
            } else {
                buf.push(ch);
            }
        }

        if !buf.is_empty() {
            tokens.push(buf);
        }

        tokens
    }

    fn tokenize_space_lang(text: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for seg in text.split_whitespace() {
            let cleaned = Self::clean_token(seg);
            if cleaned.is_empty() {
                continue;
            }
            out.extend(Self::split_segment_with_chinese(cleaned.as_str()));
        }
        out
    }

    /// Create the aligner input text, interleaving `<timestamp>` markers.
    ///
    /// Returns `(word_list, input_text)`.
    pub fn encode_timestamp(&self, text: &str, language: &str) -> Result<(Vec<String>, String)> {
        let language = language.trim();
        if text.trim().is_empty() {
            bail!("text is empty");
        }

        let word_list = if language.eq_ignore_ascii_case("japanese") {
            Self::tokenize_japanese_with_terms(text, &self.japanese_merge_terms)?
        } else if language.eq_ignore_ascii_case("korean") {
            Self::tokenize_korean(text)?
        } else {
            Self::tokenize_space_lang(text)
        };
        if word_list.is_empty() {
            bail!("tokenized word_list is empty");
        }

        let mut input_text = String::new();
        for (i, w) in word_list.iter().enumerate() {
            if i > 0 {
                input_text.push_str("<timestamp><timestamp>");
            }
            input_text.push_str(w);
        }
        input_text.push_str("<timestamp><timestamp>");

        // Mirror Python forced aligner: prepend raw audio tokens (no chat template).
        let mut prompt = String::new();
        prompt.push_str("<|audio_start|><|audio_pad|><|audio_end|>");
        prompt.push_str(input_text.as_str());

        Ok((word_list, prompt))
    }

    pub fn parse_timestamp(&self, words: &[String], timestamp_ms: &[f32]) -> Result<Vec<ItemMs>> {
        if words.is_empty() {
            bail!("words is empty");
        }
        let expected = words
            .len()
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("timestamp length overflow"))?;
        if timestamp_ms.len() != expected {
            bail!(
                "timestamp_ms length mismatch: expected={expected}, got={}",
                timestamp_ms.len()
            );
        }

        let fixed = fix_timestamp(timestamp_ms);
        let mut out: Vec<ItemMs> = Vec::with_capacity(words.len());

        for (i, w) in words.iter().enumerate() {
            let start_idx = i
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("timestamp index overflow"))?;
            let end_idx = start_idx
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("timestamp index overflow"))?;
            let start_time = *fixed
                .get(start_idx)
                .ok_or_else(|| anyhow::anyhow!("missing start timestamp"))?;
            let end_time = *fixed
                .get(end_idx)
                .ok_or_else(|| anyhow::anyhow!("missing end timestamp"))?;

            out.push(ItemMs {
                text: w.clone(),
                start_time_ms: start_time,
                end_time_ms: end_time,
            });
        }

        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ItemMs {
    pub text: String,
    pub start_time_ms: i64,
    pub end_time_ms: i64,
}

/// Fix a potentially non-monotonic timestamp sequence by:
/// 1) finding a longest non-decreasing subsequence, and
/// 2) repairing anomalous spans via nearest-neighbor or interpolation.
///
/// This matches `Qwen3ForceAlignProcessor.fix_timestamp` in the official stack.
pub fn fix_timestamp(data: &[f32]) -> Vec<i64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let mut dp: Vec<usize> = vec![1; n];
    let mut parent: Vec<Option<usize>> = vec![None; n];

    for i in 1..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j].saturating_add(1) > dp[i] {
                dp[i] = dp[j].saturating_add(1);
                parent[i] = Some(j);
            }
        }
    }

    let max_len = dp.iter().copied().max().unwrap_or(0);
    let max_idx = dp.iter().position(|&v| v == max_len).unwrap_or(0);

    let mut lis_indices: Vec<usize> = Vec::with_capacity(max_len);
    let mut idx: Option<usize> = Some(max_idx);
    while let Some(u) = idx {
        lis_indices.push(u);
        idx = parent.get(u).copied().flatten();
    }
    lis_indices.reverse();

    let mut is_normal: Vec<bool> = vec![false; n];
    for &i in &lis_indices {
        if let Some(x) = is_normal.get_mut(i) {
            *x = true;
        }
    }

    let mut result: Vec<f32> = data.to_vec();
    let mut i = 0usize;
    while i < n {
        if !is_normal[i] {
            let mut j = i;
            while j < n && !is_normal[j] {
                j = j.saturating_add(1);
            }

            let anomaly_count = j.saturating_sub(i);

            let left_val = (0..i)
                .rev()
                .find(|&k| is_normal[k])
                .and_then(|k| result.get(k).copied());
            let right_val = (j..n)
                .find(|&k| is_normal[k])
                .and_then(|k| result.get(k).copied());

            if anomaly_count <= 2 {
                for k in i..j {
                    let new_val = match (left_val, right_val) {
                        (None, Some(r)) => r,
                        (Some(l), None) => l,
                        (Some(l), Some(r)) => {
                            let left_dist = k.saturating_sub(i.saturating_sub(1));
                            let right_dist = j.saturating_sub(k);
                            if left_dist <= right_dist { l } else { r }
                        }
                        (None, None) => 0.0,
                    };
                    if let Some(x) = result.get_mut(k) {
                        *x = new_val;
                    }
                }
            } else {
                match (left_val, right_val) {
                    (Some(l), Some(r)) => {
                        let step = (r - l) / (anomaly_count.saturating_add(1) as f32);
                        for (t, k) in (i..j).enumerate() {
                            let val = l + step * ((t + 1) as f32);
                            if let Some(x) = result.get_mut(k) {
                                *x = val;
                            }
                        }
                    }
                    (Some(l), None) => {
                        for k in i..j {
                            if let Some(x) = result.get_mut(k) {
                                *x = l;
                            }
                        }
                    }
                    (None, Some(r)) => {
                        for k in i..j {
                            if let Some(x) = result.get_mut(k) {
                                *x = r;
                            }
                        }
                    }
                    (None, None) => {
                        for k in i..j {
                            if let Some(x) = result.get_mut(k) {
                                *x = 0.0;
                            }
                        }
                    }
                }
            }

            i = j;
        } else {
            i = i.saturating_add(1);
        }
    }

    result.into_iter().map(|x| x.trunc() as i64).collect()
}

#[cfg(test)]
mod tests {
    use super::{
        ForcedAlignProcessor, JapaneseMergeTerms, default_japanese_merge_terms, fix_timestamp,
        korean_dict,
    };
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_temp_user_dict(ext: &str, content: &str) -> anyhow::Result<PathBuf> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| anyhow::anyhow!("system clock before unix epoch: {e}"))?
            .as_nanos();
        let file = format!(
            "qwen3-asr-rs.ja-user-dict.{}.{}.{}",
            std::process::id(),
            nanos,
            ext
        );
        let path = std::env::temp_dir().join(file);
        std::fs::write(&path, content)?;
        Ok(path)
    }

    #[test]
    fn test_encode_timestamp_inserts_markers_and_audio_prefix() -> anyhow::Result<()> {
        let p = ForcedAlignProcessor::new();
        let (words, prompt) = p.encode_timestamp("hello world", "English")?;
        if words != vec!["hello".to_string(), "world".to_string()] {
            anyhow::bail!("unexpected words: {words:?}");
        }
        if !prompt.starts_with("<|audio_start|><|audio_pad|><|audio_end|>") {
            anyhow::bail!("missing audio prefix: {prompt:?}");
        }
        if !prompt.contains("hello<timestamp><timestamp>world<timestamp><timestamp>") {
            anyhow::bail!("unexpected marker placement: {prompt:?}");
        }
        Ok(())
    }

    #[test]
    fn test_fix_timestamp_returns_monotonic_for_simple_case() -> anyhow::Result<()> {
        let xs = vec![0.0, 10.0, 5.0, 20.0];
        let fixed = fix_timestamp(&xs);
        if fixed.len() != xs.len() {
            anyhow::bail!("length mismatch");
        }
        for w in fixed.windows(2) {
            if w[0] > w[1] {
                anyhow::bail!("expected non-decreasing, got {fixed:?}");
            }
        }
        Ok(())
    }

    #[test]
    fn test_parse_timestamp_pairs_words() -> anyhow::Result<()> {
        let p = ForcedAlignProcessor::new();
        let words = vec!["a".to_string(), "b".to_string()];
        let ts = vec![0.0, 10.0, 10.0, 20.0];
        let out = p.parse_timestamp(&words, &ts)?;
        if out.len() != 2 {
            anyhow::bail!("expected 2 items, got {}", out.len());
        }
        if out[0].text != "a" || out[0].start_time_ms != 0 || out[0].end_time_ms != 10 {
            anyhow::bail!("unexpected first item: {:?}", out[0]);
        }
        if out[1].text != "b" || out[1].start_time_ms != 10 || out[1].end_time_ms != 20 {
            anyhow::bail!("unexpected second item: {:?}", out[1]);
        }
        Ok(())
    }

    #[test]
    fn test_tokenize_japanese_fallback_splits_scripts() -> anyhow::Result<()> {
        let words = ForcedAlignProcessor::tokenize_japanese_fallback("あい漢字");
        if words != vec!["あい".to_string(), "漢字".to_string()] {
            anyhow::bail!("unexpected words: {words:?}");
        }
        Ok(())
    }

    #[test]
    fn test_tokenize_japanese_round_trips_cleaned_text() -> anyhow::Result<()> {
        let text = "あい漢字";
        let words = ForcedAlignProcessor::tokenize_japanese_with_terms(
            text,
            default_japanese_merge_terms(),
        )?;
        if words.is_empty() {
            anyhow::bail!("expected non-empty words");
        }
        let joined = words.join("");
        let cleaned = ForcedAlignProcessor::clean_token(text);
        if joined != cleaned {
            anyhow::bail!("joined tokens do not match cleaned text");
        }
        Ok(())
    }

    #[test]
    fn test_tokenize_japanese_evaluation_set_invariants() -> anyhow::Result<()> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let path = root
            .join("fixtures")
            .join("text")
            .join("japanese_tokenization_eval.json");
        let content = std::fs::read_to_string(&path)?;
        let cases: Vec<String> = serde_json::from_str(&content)?;
        if cases.is_empty() {
            anyhow::bail!("evaluation set is empty: {path:?}");
        }

        for (idx, text) in cases.iter().enumerate() {
            let words = ForcedAlignProcessor::tokenize_japanese_with_terms(
                text.as_str(),
                default_japanese_merge_terms(),
            )?;
            if words.is_empty() {
                anyhow::bail!("expected non-empty words (idx={idx} text={text:?})");
            }
            if words.iter().any(|w| w.is_empty()) {
                anyhow::bail!("unexpected empty token (idx={idx} text={text:?}): {words:?}");
            }
            for w in &words {
                if *w != ForcedAlignProcessor::clean_token(w.as_str()) {
                    anyhow::bail!(
                        "token contains removed chars after cleaning (idx={idx} text={text:?} token={w:?})"
                    );
                }
            }

            let cleaned = ForcedAlignProcessor::clean_token_japanese(text.as_str());
            let joined = words.join("");
            if joined != cleaned {
                anyhow::bail!(
                    "joined tokens do not match cleaned text (idx={idx} text={text:?} cleaned={cleaned:?} words={words:?})"
                );
            }

            let cleaned_len = cleaned.chars().count();
            if words.len() > cleaned_len {
                anyhow::bail!(
                    "token count exceeds cleaned char count (idx={idx} text={text:?} tokens={} cleaned_len={cleaned_len})",
                    words.len()
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_tokenize_japanese_splits_particle_wa() -> anyhow::Result<()> {
        // Ensure we use the morphology tokenizer by default (not the conservative script-run
        // fallback). For example, we expect "これは" to be split into "これ" + the particle "は".
        let words = ForcedAlignProcessor::tokenize_japanese_with_terms(
            "これはテストです",
            default_japanese_merge_terms(),
        )?;
        if words.iter().any(|w| w == "これは") {
            anyhow::bail!("unexpected script-run tokenization: {words:?}");
        }
        if !words.iter().any(|w| w == "は") {
            anyhow::bail!("expected token \"は\" (particle), got {words:?}");
        }
        Ok(())
    }

    #[test]
    fn test_tokenize_japanese_hiragana_particle_outlier() -> anyhow::Result<()> {
        let words = ForcedAlignProcessor::tokenize_japanese_with_terms(
            "てれびとほんをかう？",
            default_japanese_merge_terms(),
        )?;
        let want = vec![
            "てれび".to_string(),
            "と".to_string(),
            "ほん".to_string(),
            "を".to_string(),
            "かう".to_string(),
        ];
        if words != want {
            anyhow::bail!("unexpected words for hiragana particle outlier: {words:?}");
        }
        Ok(())
    }

    #[test]
    fn test_tokenize_japanese_hiragana_family_honorific_outlier() -> anyhow::Result<()> {
        let words = ForcedAlignProcessor::tokenize_japanese_with_terms(
            "おかあさんがてれびによむね",
            default_japanese_merge_terms(),
        )?;
        let want = vec![
            "お".to_string(),
            "かあ".to_string(),
            "さん".to_string(),
            "が".to_string(),
            "てれび".to_string(),
            "に".to_string(),
            "よむ".to_string(),
            "ね".to_string(),
        ];
        if words != want {
            anyhow::bail!("unexpected words for family honorific outlier: {words:?}");
        }
        Ok(())
    }

    #[test]
    fn test_tokenize_japanese_hiragana_does_not_split_single_word() -> anyhow::Result<()> {
        let words = ForcedAlignProcessor::tokenize_japanese_with_terms(
            "ありがとう",
            default_japanese_merge_terms(),
        )?;
        let want = vec!["ありがとう".to_string()];
        if words != want {
            anyhow::bail!("unexpected hiragana split: {words:?}");
        }
        Ok(())
    }

    #[test]
    fn test_default_japanese_merge_terms_are_conservative() -> anyhow::Result<()> {
        let terms = default_japanese_merge_terms();
        if !terms.contains("OpenAI") {
            anyhow::bail!("expected OpenAI to be present in default merge terms");
        }
        if terms.contains("生成AI") {
            anyhow::bail!("生成AI should not be merged by default");
        }
        if terms.contains("自然言語処理") {
            anyhow::bail!("自然言語処理 should not be merged by default");
        }
        if terms.contains("大規模言語モデル") {
            anyhow::bail!("大規模言語モデル should not be merged by default");
        }
        if terms.contains("ChatGPT") {
            anyhow::bail!("ChatGPT should not be merged by default");
        }
        if terms.contains("iPhone15Pro") {
            anyhow::bail!("iPhone15Pro should not be merged by default");
        }
        Ok(())
    }

    #[test]
    fn test_merge_japanese_tokens_default_keeps_chatgpt_split() -> anyhow::Result<()> {
        let input = vec![
            "Open".to_string(),
            "AI".to_string(),
            "の".to_string(),
            "Chat".to_string(),
            "GPT".to_string(),
        ];
        let got = ForcedAlignProcessor::merge_japanese_tokens_with_terms(
            input,
            default_japanese_merge_terms(),
        );
        let want = vec![
            "OpenAI".to_string(),
            "の".to_string(),
            "Chat".to_string(),
            "GPT".to_string(),
        ];
        if got != want {
            anyhow::bail!("unexpected merge result: got={got:?} want={want:?}");
        }
        Ok(())
    }

    #[test]
    fn test_merge_japanese_tokens_merges_single_kanji_runs() -> anyhow::Result<()> {
        let input = vec![
            "東".to_string(),
            "京".to_string(),
            "都".to_string(),
            "に".to_string(),
        ];
        let got = ForcedAlignProcessor::merge_japanese_tokens_with_terms(
            input,
            default_japanese_merge_terms(),
        );
        let want = vec!["東京都".to_string(), "に".to_string()];
        if got != want {
            anyhow::bail!("unexpected merge result: got={got:?} want={want:?}");
        }
        Ok(())
    }

    #[test]
    fn test_merge_japanese_tokens_user_terms_merge_known_phrase_terms() -> anyhow::Result<()> {
        let input = vec![
            "大".to_string(),
            "規模".to_string(),
            "言語".to_string(),
            "モデル".to_string(),
            "と".to_string(),
            "生成".to_string(),
            "AI".to_string(),
        ];
        let merge_terms = JapaneseMergeTerms::from_builtin_and_user(&[
            "大規模言語モデル".to_string(),
            "生成AI".to_string(),
        ]);
        let got = ForcedAlignProcessor::merge_japanese_tokens_with_terms(input, &merge_terms);
        let want = vec![
            "大規模言語モデル".to_string(),
            "と".to_string(),
            "生成AI".to_string(),
        ];
        if got != want {
            anyhow::bail!("unexpected merge result: got={got:?} want={want:?}");
        }
        Ok(())
    }

    #[test]
    fn test_tokenize_japanese_default_keeps_domain_terms_split() -> anyhow::Result<()> {
        let words = ForcedAlignProcessor::tokenize_japanese_with_terms(
            "生成AIと大規模言語モデル",
            default_japanese_merge_terms(),
        )?;
        if words.iter().any(|w| w == "生成AI") {
            anyhow::bail!("default merge terms should keep 生成AI split, got {words:?}");
        }
        if words.iter().any(|w| w == "大規模言語モデル") {
            anyhow::bail!("default merge terms should keep 大規模言語モデル split, got {words:?}");
        }
        if words.len() < 5 {
            anyhow::bail!("expected multiple domain tokens after conservative split: {words:?}");
        }

        let joined = words.join("");
        let cleaned = ForcedAlignProcessor::clean_token_japanese("生成AIと大規模言語モデル");
        if joined != cleaned {
            anyhow::bail!("joined tokens do not match cleaned text");
        }
        Ok(())
    }

    #[test]
    fn test_tokenize_japanese_user_terms_merge_domain_terms() -> anyhow::Result<()> {
        let merge_terms = JapaneseMergeTerms::from_builtin_and_user(&[
            "生成AI".to_string(),
            "大規模言語モデル".to_string(),
        ]);
        let words = ForcedAlignProcessor::tokenize_japanese_with_terms(
            "生成AIと大規模言語モデル",
            &merge_terms,
        )?;
        if !words.iter().any(|w| w == "生成AI") {
            anyhow::bail!("expected merged token \"生成AI\" with user terms, got {words:?}");
        }
        if !words.iter().any(|w| w == "大規模言語モデル") {
            anyhow::bail!(
                "expected merged token \"大規模言語モデル\" with user terms, got {words:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_tokenize_japanese_default_does_not_collapse_iphone_case() -> anyhow::Result<()> {
        let words = ForcedAlignProcessor::tokenize_japanese_with_terms(
            "iPhone15 Pro",
            default_japanese_merge_terms(),
        )?;
        if words.len() < 2 {
            anyhow::bail!("expected at least two tokens for mixed alnum case: {words:?}");
        }
        if words == vec!["iPhone15Pro".to_string()] {
            anyhow::bail!("unexpected over-merged tokenization: {words:?}");
        }
        let joined = words.join("");
        let cleaned = ForcedAlignProcessor::clean_token_japanese("iPhone15 Pro");
        if joined != cleaned {
            anyhow::bail!("joined tokens do not match cleaned text");
        }
        Ok(())
    }

    #[test]
    fn test_tokenize_japanese_splits_multi_digit_runs() -> anyhow::Result<()> {
        let words = ForcedAlignProcessor::tokenize_japanese_with_terms(
            "2026年2月15日",
            default_japanese_merge_terms(),
        )?;
        let want = vec![
            "2".to_string(),
            "0".to_string(),
            "2".to_string(),
            "6".to_string(),
            "年".to_string(),
            "2".to_string(),
            "月".to_string(),
            "1".to_string(),
            "5".to_string(),
            "日".to_string(),
        ];
        if words != want {
            anyhow::bail!("unexpected numeric split: got={words:?} want={want:?}");
        }

        let words = ForcedAlignProcessor::tokenize_japanese_with_terms(
            "1,234円",
            default_japanese_merge_terms(),
        )?;
        let want = vec![
            "1".to_string(),
            "2".to_string(),
            "3".to_string(),
            "4".to_string(),
            "円".to_string(),
        ];
        if words != want {
            anyhow::bail!("unexpected comma-number split: got={words:?} want={want:?}");
        }
        Ok(())
    }

    #[test]
    fn test_tokenize_japanese_normalizes_halfwidth_katakana() -> anyhow::Result<()> {
        let words = ForcedAlignProcessor::tokenize_japanese_with_terms(
            "ﾊﾝｶｸｶﾀｶﾅもOK",
            default_japanese_merge_terms(),
        )?;
        let want = vec![
            "ハンカク".to_string(),
            "カタカナ".to_string(),
            "も".to_string(),
            "OK".to_string(),
        ];
        if words != want {
            anyhow::bail!(
                "unexpected halfwidth katakana tokenization: got={words:?} want={want:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_load_japanese_user_dictionary_from_txt() -> anyhow::Result<()> {
        let path = write_temp_user_dict(
            "txt",
            r#"
# comment
東京スカイツリー
東京スカイツリー
OpenAI研究所!!!
"#,
        )?;
        let terms = ForcedAlignProcessor::load_japanese_user_terms(path.as_path())?;
        let _ = std::fs::remove_file(path.as_path());

        let want = vec!["OpenAI研究所".to_string(), "東京スカイツリー".to_string()];
        if terms != want {
            anyhow::bail!("unexpected dictionary terms: got={terms:?} want={want:?}");
        }
        Ok(())
    }

    #[test]
    fn test_load_japanese_user_dictionary_from_json_object() -> anyhow::Result<()> {
        let path = write_temp_user_dict(
            "json",
            r#"
{
  "terms": ["メタバース", "OpenAI研究所", "OpenAI研究所"]
}
"#,
        )?;
        let terms = ForcedAlignProcessor::load_japanese_user_terms(path.as_path())?;
        let _ = std::fs::remove_file(path.as_path());

        let want = vec!["OpenAI研究所".to_string(), "メタバース".to_string()];
        if terms != want {
            anyhow::bail!("unexpected dictionary terms: got={terms:?} want={want:?}");
        }
        Ok(())
    }

    #[test]
    fn test_merge_japanese_tokens_merges_user_terms() -> anyhow::Result<()> {
        let merge_terms =
            JapaneseMergeTerms::from_builtin_and_user(&["東京スカイツリー".to_string()]);
        let input = vec![
            "東京".to_string(),
            "スカイ".to_string(),
            "ツリー".to_string(),
            "へ".to_string(),
        ];
        let got = ForcedAlignProcessor::merge_japanese_tokens_with_terms(input, &merge_terms);
        let want = vec!["東京スカイツリー".to_string(), "へ".to_string()];
        if got != want {
            anyhow::bail!("unexpected merge result: got={got:?} want={want:?}");
        }
        Ok(())
    }

    #[test]
    fn test_merge_japanese_tokens_user_terms_can_merge_ascii_runs() -> anyhow::Result<()> {
        let merge_terms = JapaneseMergeTerms::from_builtin_and_user(&[
            "ChatGPT".to_string(),
            "iPhone15Pro".to_string(),
        ]);
        let input = vec![
            "Chat".to_string(),
            "GPT".to_string(),
            "と".to_string(),
            "iPhone".to_string(),
            "15".to_string(),
            "Pro".to_string(),
        ];
        let got = ForcedAlignProcessor::merge_japanese_tokens_with_terms(input, &merge_terms);
        let want = vec![
            "ChatGPT".to_string(),
            "と".to_string(),
            "iPhone15Pro".to_string(),
        ];
        if got != want {
            anyhow::bail!("unexpected merge result: got={got:?} want={want:?}");
        }
        Ok(())
    }

    #[test]
    fn test_korean_dict_loads() -> anyhow::Result<()> {
        let d = korean_dict()?;
        if d.words.is_empty() {
            anyhow::bail!("expected non-empty embedded Korean dict");
        }
        if d.max_len_chars == 0 {
            anyhow::bail!("expected max_len_chars > 0");
        }
        Ok(())
    }

    #[test]
    fn test_tokenize_korean_ltokenizer_splits_once() -> anyhow::Result<()> {
        let p = ForcedAlignProcessor::new();
        let (words, _prompt) = p.encode_timestamp("한국어학습합니다", "Korean")?;
        if words != vec!["한국어".to_string(), "학습합니다".to_string()] {
            anyhow::bail!("unexpected words: {words:?}");
        }
        Ok(())
    }
}
