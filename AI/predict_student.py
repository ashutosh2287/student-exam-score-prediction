# Simple prediction script for individual student scores
from main_prediction_model import StudentScorePredictor

def predict_individual_student():
    """Interactive script to predict individual student scores"""
    
    print("=== Student Exam Score Predictor ===\n")
    
    # Initialize and train model
    predictor = StudentScorePredictor()
    print("Creating dataset and training model...")
    predictor.create_dataset(1000)
    predictor.train_model('linear')
    
    print("\nModel ready! Enter student details for prediction:\n")
    
    while True:
        try:
            # Get user input
            study_hours = float(input("Study hours per week (0.5-12): "))
            attendance = float(input("Attendance percentage (40-100): "))
            sleep_hours = float(input("Sleep hours per night (4-10): "))
            previous_score = float(input("Previous exam score (30-100): "))
            
            print("\nSelect gender:")
            print("1. Male")
            print("2. Female")
            gender_choice = input("Enter choice (1 or 2): ")
            gender = "Male" if gender_choice == "1" else "Female"
            
            print("\nSelect parental education:")
            print("1. High School")
            print("2. Bachelor")
            print("3. Master")
            print("4. PhD")
            edu_choice = input("Enter choice (1-4): ")
            education_map = {"1": "High School", "2": "Bachelor", "3": "Master", "4": "PhD"}
            parental_education = education_map.get(edu_choice, "Bachelor")
            
            print("\nTest preparation:")
            print("1. None")
            print("2. Completed")
            prep_choice = input("Enter choice (1 or 2): ")
            test_prep = "None" if prep_choice == "1" else "Completed"
            
            # Make prediction
            predicted_score = predictor.predict_score(
                study_hours, attendance, sleep_hours, previous_score,
                gender, parental_education, test_prep
            )
            
            print(f"\n🎯 PREDICTED EXAM SCORE: {predicted_score:.1f} points")
            print(f"Score Range: {predicted_score-6.6:.1f} - {predicted_score+6.6:.1f} (±6.6 typical error)")
            
            # Ask for another prediction
            another = input("\nPredict for another student? (y/n): ").lower()
            if another != 'y':
                break
                
        except ValueError:
            print("Please enter valid numbers.")
        except Exception as e:
            print(f"Error: {e}")
    
    print("Thank you for using the Student Score Predictor!")

if __name__ == "__main__":
    predict_individual_student()