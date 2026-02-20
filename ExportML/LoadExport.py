import joblib
import pandas as pd
import os

# Obtener el path relativo a este archivo
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'regression_model.joblib')

# Cargar el modelo
jModel = joblib.load(model_path)

print("Introduce los valores del estudiante:")

# Debido a OrdinalEncoder (Pipeline Pre-processing)
academic_level_mapping = {
    0: 'High School',
    1: 'Undergraduate',
    2: 'Graduate',
    3: 'Postgraduate'
}

# Colección
student_id = int(input("Student ID (ej: 9999): "))
age = int(input("Age (Rango recomendado: 16-25): "))
study_hours = float(input("Study Hours (Rango recomendado: 0.0-11.84): "))
self_study_hours = float(input("Self-Study Hours (Rango recomendado: 0.0-7.41): "))
online_classes_hours = float(input("Online Classes Hours (Rango recomendado: 0.0-6.0): "))
social_media_hours = float(input("Social Media Hours (Rango recomendado: 0.0-8.28): "))
gaming_hours = float(input("Gaming Hours (Rango recomendado: 0.0-5.64): "))
sleep_hours = float(input("Sleep Hours (Rango recomendado: 4.0-10.0): "))
screen_time_hours = float(input("Screen Time Hours (Rango recomendado: 1.0-15.30): "))
exercise_minutes = int(input("Exercise Minutes (Rango recomendado: 0-149): "))
caffeine_intake_mg = int(input("Caffeine Intake (mg) (Rango recomendado: 0-499): "))
part_time_job = int(input("Part-time Job (0 - No, 1 - Yes): "))
upcoming_deadline = int(input("Upcoming Deadline (0 - No, 1 - Yes): "))
mental_health_score = int(input("Mental Health Score (Rango recomendado: 1-10): "))
focus_index = float(input("Focus Index (Rango recomendado: 1.0-63.48): "))
burnout_level = float(input("Burnout Level (Rango recomendado: 1.0-97.58): "))
productivity_score = float(input("Productivity Score (Rango recomendado: 1.0-98.02): "))
gender = input("Gender (Male, Female, Other): ")

# Numeric, convertir a String
academic_level_num = int(input("Academic Level (0: High School, 1: Undergraduate, 2: Graduate, 3: Postgraduate): "))
academic_level = academic_level_mapping.get(academic_level_num, 'Unknown') # Default to 'Unknown'

internet_quality = input("Internet Quality (Good, Average, Poor): ")

new_data_point = pd.DataFrame({
    'student_id': [student_id],
    'age': [age],
    'study_hours': [study_hours],
    'self_study_hours': [self_study_hours],
    'online_classes_hours': [online_classes_hours],
    'social_media_hours': [social_media_hours],
    'gaming_hours': [gaming_hours],
    'sleep_hours': [sleep_hours],
    'screen_time_hours': [screen_time_hours],
    'exercise_minutes': [exercise_minutes],
    'caffeine_intake_mg': [caffeine_intake_mg],
    'part_time_job': [part_time_job],
    'upcoming_deadline': [upcoming_deadline],
    'mental_health_score': [mental_health_score],
    'focus_index': [focus_index],
    'burnout_level': [burnout_level],
    'productivity_score': [productivity_score],
    'gender': [gender],
    'academic_level': [academic_level],
    'internet_quality': [internet_quality]
})

print("\nInput data for prediction:")
print(new_data_point)

pred = jModel.predict(new_data_point)
print(f"\nPredicted exam_score: {pred[0]:.2f}")