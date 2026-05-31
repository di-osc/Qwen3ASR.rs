use anyhow::{Result, bail};

#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    pub language: Option<String>,
    pub context: String,
    pub max_new_tokens: usize,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            context: String::new(),
            max_new_tokens: 256,
        }
    }
}

impl TranscribeOptions {
    pub fn validate(&self) -> Result<()> {
        if self.max_new_tokens == 0 {
            bail!("max_new_tokens must be greater than zero");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: Option<String>,
    pub raw: String,
}

#[cfg(test)]
mod tests {
    use super::TranscribeOptions;

    #[test]
    fn default_max_new_tokens_is_256() {
        let opts = TranscribeOptions::default();
        assert_eq!(opts.max_new_tokens, 256);
    }

    #[test]
    fn rejects_zero_max_new_tokens() {
        let opts = TranscribeOptions {
            max_new_tokens: 0,
            ..TranscribeOptions::default()
        };
        let err = opts.validate().unwrap_err().to_string();
        assert!(err.contains("max_new_tokens must be greater than zero"));
    }
}
