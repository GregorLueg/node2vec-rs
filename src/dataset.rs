use burn::data::dataset::Dataset;

/// Dataset wrapper for random walk sequences
///
/// ### Fields
///
/// * `walks` - Slice of the underlying walks
pub struct WalkDataset {
    walks: Vec<Vec<u32>>,
}

#[allow(dead_code)]
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

    /// Getter function
    ///
    /// ### Params
    ///
    /// * `index` - Get the walks at this index
    fn get(&self, index: usize) -> Option<Vec<u32>> {
        self.walks.get(index).cloned()
    }

    /// Get the length
    ///
    /// ### Returns
    ///
    /// Length of the walks
    fn len(&self) -> usize {
        self.walks.len()
    }

    /// Is empty helper
    ///
    /// ### Returns
    ///
    /// Boolean indicating if walks are empty
    fn is_empty(&self) -> bool {
        self.walks.is_empty()
    }
}

impl Dataset<Vec<u32>> for WalkDataset {
    /// Getter function for a walk
    ///
    /// ### Params
    ///
    /// * `index` - The index of the walk
    ///
    /// ### Returns
    ///
    /// The given random walk
    fn get(&self, index: usize) -> Option<Vec<u32>> {
        self.walks.get(index).cloned()
    }

    /// Get the total length
    ///
    /// ### Returns
    ///
    /// Length of the random walks
    fn len(&self) -> usize {
        self.walks.len()
    }
}

#[cfg(test)]
mod dataset_tests {
    use crate::dataset::WalkDataset;

    #[test]
    fn test_dataset_get() {
        let walks = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let dataset = WalkDataset::new(walks.clone());

        assert_eq!(dataset.get(0), Some(vec![1, 2, 3]));
        assert_eq!(dataset.get(1), Some(vec![4, 5, 6]));
        assert_eq!(dataset.get(2), Some(vec![7, 8, 9]));
        assert_eq!(dataset.get(3), None);
    }

    #[test]
    fn test_dataset_len() {
        let walks = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let dataset = WalkDataset::new(walks);

        assert_eq!(dataset.len(), 3);
    }

    #[test]
    fn test_empty_dataset() {
        let dataset = WalkDataset::new(vec![]);

        assert_eq!(dataset.len(), 0);
        assert!(dataset.is_empty());
        assert_eq!(dataset.get(0), None);
    }

    #[test]
    fn test_single_walk_dataset() {
        let walks = vec![vec![1, 2, 3, 4, 5]];
        let dataset = WalkDataset::new(walks);

        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.get(0), Some(vec![1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_varying_walk_lengths() {
        let walks = vec![vec![1], vec![2, 3], vec![4, 5, 6], vec![7, 8, 9, 10]];
        let dataset = WalkDataset::new(walks.clone());

        assert_eq!(dataset.len(), 4);
        for (i, walk) in walks.iter().enumerate() {
            assert_eq!(dataset.get(i), Some(walk.clone()));
        }
    }

    #[test]
    fn test_out_of_bounds_access() {
        let walks = vec![vec![1, 2, 3]];
        let dataset = WalkDataset::new(walks);

        assert_eq!(dataset.get(1), None);
        assert_eq!(dataset.get(100), None);
    }

    #[test]
    fn test_dataset_cloning_independence() {
        let walks = vec![vec![1, 2, 3]];
        let dataset = WalkDataset::new(walks);

        let mut walk1 = dataset.get(0).unwrap();
        walk1.push(999);

        let walk2 = dataset.get(0).unwrap();
        assert_eq!(walk2, vec![1, 2, 3]); // Original unchanged
        assert_ne!(walk1, walk2);
    }

    #[test]
    fn test_large_dataset() {
        let walks: Vec<Vec<u32>> = (0..10000).map(|i| vec![i, i + 1, i + 2]).collect();
        let dataset = WalkDataset::new(walks);

        assert_eq!(dataset.len(), 10000);
        assert_eq!(dataset.get(5000), Some(vec![5000, 5001, 5002]));
        assert_eq!(dataset.get(9999), Some(vec![9999, 10000, 10001]));
    }
}
