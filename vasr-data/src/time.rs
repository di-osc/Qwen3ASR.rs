use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DurationMs(pub u64);

impl DurationMs {
    pub fn seconds(self) -> f64 {
        self.0 as f64 / 1000.0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SampleIndex(pub u64);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DurationMs,
    pub end: DurationMs,
}

impl TimeRange {
    pub fn new(start: DurationMs, end: DurationMs) -> Self {
        Self { start, end }
    }

    pub fn duration(self) -> DurationMs {
        DurationMs(self.end.0.saturating_sub(self.start.0))
    }

    pub fn overlaps(&self, other: &TimeRange) -> bool {
        self.start < other.end && other.start < self.end
    }

    pub fn contains(&self, point: DurationMs) -> bool {
        self.start <= point && point < self.end
    }
}
