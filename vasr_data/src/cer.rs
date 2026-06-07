//! Character Error Rate (CER) metrics for ASR evaluation.
//!
//! CER = (S + D + I) / N
//!
//! - S: substitutions
//! - D: deletions
//! - I: insertions
//! - N: number of characters in the reference text

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CerStats {
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
    pub reference_chars: usize,
}

impl CerStats {
    pub fn edits(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }

    pub fn cer(&self) -> f64 {
        if self.reference_chars == 0 {
            if self.edits() == 0 {
                return 0.0;
            }
            return 1.0;
        }
        self.edits() as f64 / self.reference_chars as f64
    }
}

pub fn normalize_for_cer(text: &str, remove_spaces: bool) -> String {
    text.chars()
        .filter(|ch| {
            if is_punctuation(*ch) {
                return false;
            }
            !remove_spaces || !ch.is_whitespace()
        })
        .collect()
}

fn is_punctuation(ch: char) -> bool {
    use unicode_general_category::{GeneralCategory, get_general_category};

    matches!(
        get_general_category(ch),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation
    )
}

pub fn compute_cer(reference: &str, hypothesis: &str) -> CerStats {
    let reference_chars: Vec<char> = reference.chars().collect();
    let hypothesis_chars: Vec<char> = hypothesis.chars().collect();
    let (substitutions, deletions, insertions) =
        levenshtein_ops(&reference_chars, &hypothesis_chars);
    CerStats {
        substitutions,
        deletions,
        insertions,
        reference_chars: reference_chars.len(),
    }
}

fn levenshtein_ops(reference: &[char], hypothesis: &[char]) -> (usize, usize, usize) {
    let n = reference.len();
    let m = hypothesis.len();
    if n == 0 {
        return (0, 0, m);
    }
    if m == 0 {
        return (0, n, 0);
    }

    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate().skip(1) {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate().skip(1) {
        *cell = j;
    }

    for i in 1..=n {
        for j in 1..=m {
            let cost = usize::from(reference[i - 1] != hypothesis[j - 1]);
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    let mut i = n;
    let mut j = m;
    let mut substitutions = 0;
    let mut deletions = 0;
    let mut insertions = 0;
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && reference[i - 1] == hypothesis[j - 1] {
            i -= 1;
            j -= 1;
            continue;
        }
        if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
            substitutions += 1;
            i -= 1;
            j -= 1;
            continue;
        }
        if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            deletions += 1;
            i -= 1;
            continue;
        }
        insertions += 1;
        j -= 1;
    }

    (substitutions, deletions, insertions)
}

#[cfg(test)]
mod tests {
    use super::{CerStats, compute_cer, normalize_for_cer};

    #[test]
    fn cer_formula_matches_edit_counts() {
        let stats = compute_cer("kitten", "sitting");
        assert_eq!(
            stats,
            CerStats {
                substitutions: 2,
                deletions: 0,
                insertions: 1,
                reference_chars: 6,
            }
        );
        assert!((stats.cer() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn cer_handles_chinese_characters() {
        let stats = compute_cer("你好世界", "你好世");
        assert_eq!(
            stats,
            CerStats {
                substitutions: 0,
                deletions: 1,
                insertions: 0,
                reference_chars: 4,
            }
        );
        assert!((stats.cer() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn normalize_for_cer_strips_whitespace_by_default() {
        let normalized = normalize_for_cer("你 好\n世界", true);
        assert_eq!(normalized, "你好世界");
    }

    #[test]
    fn normalize_for_cer_strips_punctuation() {
        let normalized = normalize_for_cer("你好，世界！How are you?", true);
        assert_eq!(normalized, "你好世界Howareyou");
    }
}
