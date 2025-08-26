# Student Exam Score Prediction Project

A complete machine learning project that predicts student exam scores based on study habits, attendance, sleep patterns, and other factors.

## Project Overview

This project uses machine learning algorithms to predict student exam performance based on:
- Study hours per week
- Attendance percentage  
- Sleep hours per night
- Previous exam scores
- Demographics (gender, parental education)
- Test preparation completion

## Files Included

1. **main_prediction_model.py** - Main Python script with complete ML pipeline
2. **requirements.txt** - Required Python packages
3. **README.md** - This documentation file

## Setup Instructions

### Step 1: Install Python
Make sure Python 3.8+ is installed on your computer.

### Step 2: Install Required Packages
Open terminal/command prompt and run:
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn plotly
```

### Step 3: Run the Project
Execute the main script:
```bash
python main_prediction_model.py
```

## What the Script Does

1. **Creates Dataset**: Generates 1000 realistic student records
2. **Trains Model**: Uses Linear Regression for exam score prediction
3. **Evaluates Performance**: Calculates accuracy metrics (R², MAE, MSE)
4. **Makes Predictions**: Shows example predictions for different student types
5. **Creates Visualizations**: Generates plots showing model performance
6. **Saves Results**: Outputs CSV files with data and predictions

## Output Files

After running, you'll get:
- `student_dataset.csv` - Complete student dataset
- `prediction_results.csv` - Model predictions and actual scores
- `model_analysis.png` - Visualization plots

## Model Performance

The Linear Regression model typically achieves:
- **R² Score**: ~0.62 (explains 62% of variance)
- **Mean Absolute Error**: ~6.6 points
- **Accuracy**: 75% of predictions within 10 points of actual scores

## Key Features

### Prediction Function
Use the trained model to predict scores for new students:

```python
predictor = StudentScorePredictor()
predictor.create_dataset()
predictor.train_model()

score = predictor.predict_score(
    study_hours=7,
    attendance_pct=90, 
    sleep_hours=8,
    previous_score=75,
    gender='Female',
    parental_education='Bachelor',
    test_preparation='Completed'
)
print(f"Predicted score: {score:.1f}")
```

### Key Insights

1. **Previous exam scores** are the strongest predictor (highest correlation)
2. **Study hours** significantly impact performance (~5.7 points per hour)
3. **Test preparation** provides ~1.8 point boost
4. **Sleep and attendance** have moderate positive effects
5. **Demographics** show smaller but measurable effects

## Customization

### Change Dataset Size
```python
data = predictor.create_dataset(n_samples=2000)  # Create 2000 students
```

### Try Different Models
```python
predictor.train_model('random_forest')  # Use Random Forest instead
```

### Modify Features
Edit the feature list in the `StudentScorePredictor` class to add/remove variables.

## Educational Applications

- **Early Warning System**: Identify at-risk students
- **Academic Counseling**: Data-driven advice for students
- **Resource Allocation**: Target interventions effectively
- **Performance Tracking**: Monitor improvement over time

## Technical Details

- **Algorithm**: Linear Regression with standardized features
- **Validation**: 80/20 train-test split
- **Features**: 7 input variables (4 numeric, 3 categorical)
- **Target**: Exam scores (0-100 scale)
- **Preprocessing**: Label encoding for categories, standard scaling for numeric

## Troubleshooting

### Import Errors
If you get import errors, install missing packages:
```bash
pip install [package_name]
```

### Permission Issues
On Windows, try running as administrator or use:
```bash
python -m pip install [package_name]
```

### Plot Display Issues
If plots don't show, install additional backend:
```bash
pip install tkinter
```

## Future Enhancements

- Add more student factors (socioeconomic, mental health)
- Implement deep learning models
- Create web interface for predictions
- Add temporal analysis across semesters
- Include teacher/course quality metrics

## License

This project is for educational purposes. Feel free to modify and use for learning.

## Support

For questions or issues, check:
1. Python version compatibility (3.8+)
2. Package installation success
3. File permissions in working directory

---

**Author**: Educational ML Project  
**Created**: August 2025  
**Purpose**: Student Performance Prediction