import pandas as pd
import numpy as np
import streamlit as st

# Load datasets
food_data = pd.read_csv('food.csv')
nutrition_data = pd.read_csv('nutrition_distriution.csv')

# Replace 'null' with NaN
food_data.replace('null', np.nan, inplace=True)
nutrition_data.replace('null', np.nan, inplace=True)

# Standardizing column names
food_data.columns = food_data.columns.str.strip().str.replace(' ', '_')
nutrition_data.columns = nutrition_data.columns.str.strip().str.replace(' ', '_')

# Merging data
columns_to_merge_on = ['Calories', 'Fats', 'Proteins', 'Iron', 'Calcium', 'Sodium', 'Potassium', 'Carbohydrates', 'Fibre', 'VitaminD', 'Sugars']
merged_data = pd.merge(food_data, nutrition_data, on=columns_to_merge_on, how='outer')
merged_data.fillna("Data Not Available", inplace=True)

# Assigning Meal Categories (if not present in the dataset)
if 'Meal_Type' not in merged_data.columns:
    def assign_meal_type(food_item):
        food_item = food_item.lower()
        if any(word in food_item for word in ['oats', 'pancake', 'toast', 'egg', 'smoothie', 'yogurt', 'fruit', 'cereal']):
            return 'Breakfast'
        elif any(word in food_item for word in ['salad', 'rice', 'chicken', 'fish', 'pasta', 'soup', 'sandwich']):
            return 'Lunch'
        elif any(word in food_item for word in ['dinner', 'grill', 'steak', 'roast', 'lentil', 'stew', 'curry']):
            return 'Dinner'
        else:
            return 'Anytime Snack'
    
    merged_data['Meal_Type'] = merged_data['Food_items'].apply(assign_meal_type)

# 🌟 **Streamlit UI**
st.title("🥗 Diet Recommendation System")

# User Inputs
weight = st.number_input("Enter Weight (kg):", min_value=30.0, max_value=200.0, value=70.0)
height = st.number_input("Enter Height (cm):", min_value=100.0, max_value=250.0, value=170.0)

# Function to calculate BMI category
def calculate_bmi_category(bmi):
    if bmi < 16:
        return "Severely Underweight"
    elif 16 <= bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Healthy"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Severely Overweight"

# Button to get the recommendation
if st.button("Get Recommendation"):
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    bmi_category = calculate_bmi_category(bmi)
    
    # Display BMI Category with color-coding
    if bmi_category in ["Severely Underweight", "Severely Overweight"]:
        st.error(f"### Your BMI: {bmi:.2f} ({bmi_category}) ⚠️ Consider a dietary plan.")
    elif bmi_category == "Healthy":
        st.success(f"### Your BMI: {bmi:.2f} ({bmi_category}) ✅ Keep up the good work!")
    else:
        st.warning(f"### Your BMI: {bmi:.2f} ({bmi_category}) ⚠️ Consider adjustments.")

    # 🎯 **Filter Food Recommendations by Meal Type**
    # Breakfast
    breakfast_data = merged_data[merged_data['Meal_Type'] == 'Breakfast']
    if not breakfast_data.empty:
        breakfast = breakfast_data.sample(n=min(3, len(breakfast_data)), replace=False)
        st.subheader("🍳 Breakfast Recommendations")
        st.table(breakfast[['Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates']])
    else:
        st.warning("No breakfast recommendations available.")

    # Lunch
    lunch_data = merged_data[merged_data['Meal_Type'] == 'Lunch']
    if not lunch_data.empty:
        lunch = lunch_data.sample(n=min(3, len(lunch_data)), replace=False)
        st.subheader("🍛 Lunch Recommendations")
        st.table(lunch[['Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates']])
    else:
        st.warning("No lunch recommendations available.")

    # Dinner
    dinner_data = merged_data[merged_data['Meal_Type'] == 'Dinner']
    if not dinner_data.empty:
        dinner = dinner_data.sample(n=min(3, len(dinner_data)), replace=False)
        st.subheader("🍽️ Dinner Recommendations")
        st.table(dinner[['Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates']])
    else:
        st.warning("No dinner recommendations available.")

    st.success("Enjoy your healthy diet! 😊")
