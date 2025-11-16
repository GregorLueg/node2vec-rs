use burn::data::dataset::Dataset;

/// Dataset wrapper for random walk sequences
///
/// ### Fields
///
/// * `walks` - Slice of the underlying walks
pub struct WalkDataset {
    walks: Vec<Vec<u32>>,
}

impl WalkDataset {
    /// Creates a new dataset from a collection of walks
    ///
    /// ### Params
    ///
    /// * `walks` - Vector of walks, where each walk is a sequence of node IDs
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn new(walks: Vec<Vec<u32>>) -> Self {
        Self { walks }
    }
}

impl Dataset<Vec<u32>> for WalkDataset {
    fn get(&self, index: usize) -> Option<Vec<u32>> {
        self.walks.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.walks.len()
    }
}
