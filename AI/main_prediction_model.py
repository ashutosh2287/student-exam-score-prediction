# Student Exam Score Prediction Model
# Complete Machine Learning Project

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class StudentScorePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.le_gender = LabelEncoder()
        self.le_parent_edu = LabelEncoder()
        self.le_test_prep = LabelEncoder()
        self.feature_names = [
            'study_hours_per_week',
            'attendance_percentage', 
            'sleep_hours_per_night',
            'previous_exam_score',
            'gender_encoded',
            'parental_education_encoded',
            'test_preparation_encoded'
        ]
        
    def create_dataset(self, n_samples=1000):
        """Create realistic student dataset"""
        np.random.seed(42)
        
        # Generate features
        study_hours = np.clip(np.random.normal(5.5, 2.5, n_samples), 0.5, 12)
        attendance_pct = np.clip(np.random.normal(85, 15, n_samples), 40, 100)
        sleep_hours = np.clip(np.random.normal(7, 1.5, n_samples), 4, 10)
        previous_exam_score = np.clip(np.random.normal(68.9, 15, n_samples), 30, 100)
        
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.45, 0.55])
        parental_education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                            n_samples, p=[0.3, 0.4, 0.25, 0.05])
        test_prep = np.random.choice(['None', 'Completed'], n_samples, p=[0.65, 0.35])
        
        # Create target variable with realistic relationships
        exam_score = (
            0.6 * previous_exam_score +
            2.5 * study_hours +
            0.15 * attendance_pct +
            1.0 * sleep_hours +
            np.where(test_prep == 'Completed', 5, 0) +
            np.where(gender == 'Female', 2, 0) +
            np.random.normal(0, 8, n_samples)
        )
        exam_score = np.clip(exam_score, 0, 100)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'study_hours_per_week': study_hours,
            'attendance_percentage': attendance_pct,
            'sleep_hours_per_night': sleep_hours,
            'previous_exam_score': previous_exam_score,
            'gender': gender,
            'parental_education': parental_education,
            'test_preparation': test_prep,
            'exam_score': exam_score
        })
        
        print(f"Dataset created with {n_samples} students")
        return self.data
    
    def preprocess_data(self):
        """Encode categorical variables and prepare features"""
        # Encode categorical variables
        self.data['gender_encoded'] = self.le_gender.fit_transform(self.data['gender'])
        self.data['parental_education_encoded'] = self.le_parent_edu.fit_transform(self.data['parental_education'])
        self.data['test_preparation_encoded'] = self.le_test_prep.fit_transform(self.data['test_preparation'])
        
        # Prepare features and target
        X = self.data[self.feature_names]
        y = self.data['exam_score']
        
        return X, y
    
    def train_model(self, model_type='linear'):
        """Train the prediction model"""
        X, y = self.preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'linear':
            self.model = LinearRegression()
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.metrics = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"Model Training Complete ({model_type})")
        print(f"MSE: {mse:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"R² Score: {r2:.3f}")
        
        return self.metrics
    
    def predict_score(self, study_hours, attendance_pct, sleep_hours, previous_score, 
                     gender, parental_education, test_preparation):
        """Predict exam score for a student"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Encode categorical variables
        gender_enc = self.le_gender.transform([gender])[0]
        parent_edu_enc = self.le_parent_edu.transform([parental_education])[0]
        test_prep_enc = self.le_test_prep.transform([test_preparation])[0]
        
        # Create feature array
        features = np.array([[study_hours, attendance_pct, sleep_hours, previous_score,
                             gender_enc, parent_edu_enc, test_prep_enc]])
        
        # Scale and predict
        if isinstance(self.model, LinearRegression):
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
        else:
            prediction = self.model.predict(features)[0]
        
        return max(0, min(100, prediction))
    
    def save_results(self):
        """Save predictions and dataset to CSV files"""
        # Save dataset
        self.data.to_csv('student_dataset.csv', index=False)
        
        # Save predictions
        if hasattr(self, 'metrics'):
            results_df = pd.DataFrame({
                'Actual_Score': self.metrics['y_test'],
                'Predicted_Score': self.metrics['y_pred'],
                'Absolute_Error': abs(self.metrics['y_test'] - self.metrics['y_pred'])
            })
            results_df.to_csv('prediction_results.csv', index=False)
            print("Results saved to CSV files")
    
    def plot_results(self):
        """Create visualization plots"""
        if not hasattr(self, 'metrics'):
            print("No results to plot. Train model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted scatter plot
        axes[0,0].scatter(self.metrics['y_pred'], self.metrics['y_test'], alpha=0.6)
        axes[0,0].plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
        axes[0,0].set_xlabel('Predicted Score')
        axes[0,0].set_ylabel('Actual Score')
        axes[0,0].set_title(f'Actual vs Predicted (R² = {self.metrics["R2"]:.3f})')
        axes[0,0].legend()
        
        # 2. Error distribution
        errors = self.metrics['y_test'] - self.metrics['y_pred']
        axes[0,1].hist(errors, bins=20, alpha=0.7, color='skyblue')
        axes[0,1].set_xlabel('Prediction Error')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Error Distribution')
        
        # 3. Feature correlations
        numeric_cols = ['study_hours_per_week', 'attendance_percentage', 'sleep_hours_per_night', 
                       'previous_exam_score', 'exam_score']
        corr_matrix = self.data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
        axes[1,0].set_title('Feature Correlations')
        
        # 4. Feature importance (if Random Forest)
        if isinstance(self.model, RandomForestRegressor):
            importances = self.model.feature_importances_
            axes[1,1].barh(self.feature_names, importances)
            axes[1,1].set_xlabel('Importance')
            axes[1,1].set_title('Feature Importance (Random Forest)')
        else:
            # Linear regression coefficients
            coefficients = abs(self.model.coef_)
            axes[1,1].barh(self.feature_names, coefficients)
            axes[1,1].set_xlabel('|Coefficient|')
            axes[1,1].set_title('Feature Coefficients (Linear Regression)')
        
        plt.tight_layout()
        plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Plots saved as 'model_analysis.png'")

def main():
    """Main function to run the complete prediction model"""
    print("=== Student Exam Score Prediction Model ===\n")
    
    # Initialize predictor
    predictor = StudentScorePredictor()
    
    # Create dataset
    data = predictor.create_dataset(1000)
    print(f"Dataset shape: {data.shape}")
    print("\nDataset preview:")
    print(data.head())
    
    # Train model
    print("\n=== Training Linear Regression Model ===")
    metrics = predictor.train_model('linear')
    
    # Save results
    predictor.save_results()
    
    # Make sample predictions
    print("\n=== Sample Predictions ===")
    test_students = [
        {
            'name': 'High Achiever',
            'study_hours': 8.0, 'attendance_pct': 95.0, 'sleep_hours': 8.0,
            'previous_score': 85.0, 'gender': 'Female', 
            'parental_education': 'Master', 'test_preparation': 'Completed'
        },
        {
            'name': 'Average Student', 
            'study_hours': 5.0, 'attendance_pct': 80.0, 'sleep_hours': 7.0,
            'previous_score': 70.0, 'gender': 'Male',
            'parental_education': 'Bachelor', 'test_preparation': 'None'
        },
        {
            'name': 'Struggling Student',
            'study_hours': 3.0, 'attendance_pct': 65.0, 'sleep_hours': 5.5,
            'previous_score': 45.0, 'gender': 'Male',
            'parental_education': 'High School', 'test_preparation': 'None'
        }
    ]
    
    for student in test_students:
        predicted_score = predictor.predict_score(
            student['study_hours'], student['attendance_pct'], student['sleep_hours'],
            student['previous_score'], student['gender'], student['parental_education'],
            student['test_preparation']
        )
        print(f"{student['name']}: {predicted_score:.1f} points")
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    predictor.plot_results()
    
    print("\n=== Project Complete ===")
    print("Files created:")
    print("- student_dataset.csv (complete dataset)")
    print("- prediction_results.csv (model predictions)")
    print("- model_analysis.png (visualization plots)")

if __name__ == "__main__":
    main()