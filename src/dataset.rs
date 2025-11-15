use burn::data::dataset::Dataset;

pub struct WalkDataset {
    walks: Vec<Vec<u32>>,
}

impl WalkDataset {
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
