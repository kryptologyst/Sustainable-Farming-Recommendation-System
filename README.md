# Sustainable Farming Recommendation System

An AI-powered system that recommends sustainable farming practices based on soil type, water availability, temperature, and crop type. This research demonstration tool uses machine learning to suggest eco-friendly agricultural practices such as crop rotation, organic farming, drip irrigation, and compost usage.

## 🌱 Overview

This project demonstrates the application of machine learning in sustainable agriculture by providing personalized farming recommendations. The system analyzes environmental and agricultural parameters to suggest the most appropriate sustainable farming practice for a given scenario.

### Key Features

- **Multi-Model Approach**: Implements Logistic Regression, Random Forest, XGBoost, and Neural Networks
- **Comprehensive Evaluation**: Domain-specific metrics and performance analysis
- **Interactive Demo**: Streamlit-based web interface for real-time recommendations
- **Reproducible Research**: Deterministic seeding and structured configuration
- **Modern Stack**: PyTorch, scikit-learn, XGBoost, and visualization tools

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Sustainable-Farming-Recommendation-System.git
   cd Sustainable-Farming-Recommendation-System
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models**:
   ```bash
   python scripts/train.py
   ```

4. **Launch the interactive demo**:
   ```bash
   streamlit run demo/app.py
   ```

## Data Schema

The system uses synthetic farming data with the following features:

### Input Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `soil_type` | Categorical | 0-4 | Soil type (Sandy, Loamy, Clay, Silt, Peat) |
| `water_availability` | Continuous | 0-100% | Water availability percentage |
| `avg_temperature` | Continuous | 15-40°C | Average temperature |
| `crop_type` | Categorical | 0-4 | Crop type (Wheat, Rice, Corn, Soybean, Vegetables) |

### Output Classes

| Class ID | Recommendation | Description |
|----------|----------------|-------------|
| 0 | Crop Rotation | Rotating different crops to improve soil health |
| 1 | Organic Farming | Using organic methods without synthetic chemicals |
| 2 | Drip Irrigation | Precise water delivery system for water conservation |
| 3 | Compost Usage | Using organic compost to enrich soil |

## Models

The system implements multiple machine learning algorithms:

### Baseline Models
- **Logistic Regression**: Linear baseline for multi-class classification
- **Random Forest**: Ensemble method with feature importance analysis

### Advanced Models
- **XGBoost**: Gradient boosting with optimized hyperparameters
- **Neural Network**: Deep learning model with dropout regularization

### Model Selection
Models are evaluated using:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices and classification reports
- Feature importance analysis
- Domain-specific evaluation metrics

## Training and Evaluation

### Training Command
```bash
python scripts/train.py --n-samples 1000 --output-dir assets
```

### Evaluation Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Domain Metrics**: Per-class performance, soil-type analysis
- **Visualization**: Confusion matrices, feature importance plots
- **Leaderboard**: Model comparison and ranking

### Output Artifacts
Training generates the following files in the `assets/` directory:
- `model_leaderboard.csv`: Performance comparison
- `confusion_matrices.png`: Confusion matrix visualizations
- `model_comparison.html`: Interactive model comparison
- `feature_importance.png`: Feature importance analysis
- `scaler.pkl`: Feature scaling parameters
- `best_model.pkl/.pth`: Trained model weights

## Interactive Demo

The Streamlit demo provides an intuitive interface for:

### Farm Parameter Input
- Soil type selection with descriptions
- Water availability slider (0-100%)
- Temperature range selection (15-40°C)
- Crop type selection with options

### Real-Time Analysis
- Parameter visualization with gauges and charts
- Instant recommendation generation
- Explanation of recommendation rationale
- Additional insights based on farm conditions

### Usage Instructions
1. Adjust farm parameters in the sidebar
2. Click "Get Recommendation" to generate suggestions
3. View detailed analysis and insights
4. Explore different scenarios and parameter combinations

## Testing

Run the test suite to verify system functionality:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_farming_system.py
```

### Test Coverage
- Data generation and preprocessing
- Model training and prediction
- Evaluation metrics and visualization
- Configuration loading and validation

## Configuration

The system uses YAML configuration files for easy customization:

### Data Configuration (`configs/data.yaml`)
- Sample size and train/test split ratios
- Feature ranges and distributions
- Random seed for reproducibility

### Model Configuration (`configs/data.yaml`)
- Neural network architecture and hyperparameters
- XGBoost parameters and settings
- Recommendation class mappings

### Visualization Configuration (`configs/viz.yaml`)
- Color schemes and themes
- Plot settings and layouts
- Map configurations for future geospatial features

## 📁 Project Structure

```
sustainable-farming-recommendation/
├── src/                    # Source code modules
│   ├── data.py            # Data generation and preprocessing
│   ├── models.py          # Machine learning models
│   └── evaluation.py      # Evaluation metrics and visualization
├── configs/               # Configuration files
│   ├── data.yaml         # Data and model configuration
│   └── viz.yaml          # Visualization settings
├── scripts/              # Training and utility scripts
│   └── train.py          # Main training pipeline
├── demo/                  # Interactive demo
│   └── app.py            # Streamlit application
├── tests/                 # Unit tests
│   └── test_farming_system.py
├── assets/                # Generated artifacts and outputs
├── data/                  # Data directories
│   ├── raw/              # Raw data files
│   ├── processed/         # Processed datasets
│   └── external/         # External data sources
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
├── DISCLAIMER.md         # Important limitations and warnings
└── README.md             # This file
```

## Research Applications

This system demonstrates several research concepts:

### Machine Learning in Agriculture
- Multi-class classification for farming recommendations
- Feature engineering for agricultural data
- Model comparison and evaluation methodologies

### Sustainable Agriculture
- Environmental parameter analysis
- Resource optimization strategies
- Eco-friendly farming practice recommendations

### Educational Use Cases
- AI applications in environmental science
- Sustainable development goal alignment
- Interdisciplinary research methodologies

## ⚠️ Important Limitations

**This is a research demonstration tool, NOT for operational use.**

### Key Limitations
- Uses synthetic data generated through heuristic rules
- No real-world validation on actual farming outcomes
- Does not replace professional agricultural advice
- May not account for local regulations or conditions

### Safety Considerations
- Always consult with agricultural experts for real farming decisions
- Consider local environmental regulations and restrictions
- Account for economic factors and market conditions
- Ensure compliance with sustainability standards

For detailed limitations and disclaimers, see [DISCLAIMER.md](DISCLAIMER.md).

## Contributing

Contributions are welcome for research and educational purposes:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code formatting
black src/ tests/ scripts/
ruff check src/ tests/ scripts/

# Run tests
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**kryptologyst**  
GitHub: [https://github.com/kryptologyst](https://github.com/kryptologyst)

## References

- Sustainable Agriculture Research and Education (SARE)
- Food and Agriculture Organization (FAO) guidelines
- Machine Learning for Agriculture: A Review
- Environmental Impact Assessment methodologies

## Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/kryptologyst/sustainable-farming-recommendation/issues)
- **Contact**: [kryptologyst](https://github.com/kryptologyst)

---

**Remember**: This tool is for research and educational purposes only. Always consult with agricultural professionals for real-world farming decisions.
# Sustainable-Farming-Recommendation-System
