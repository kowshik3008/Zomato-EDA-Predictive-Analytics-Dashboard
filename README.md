
# 🍽️ Zomato EDA & Predictive Analytics Dashboard

A professional, interactive, and machine-learning-powered web dashboard built with **Plotly Dash** and **Scikit-Learn** to analyze and predict restaurant ratings based on the Zomato dataset.

This project goes beyond simple Exploratory Data Analysis (EDA) by integrating predictive modeling and unsupervised clustering to provide deeper, actionable insights into the restaurant ecosystem.

---

## ✨ Features

### 📊 Advanced Interactive Visualizations
- **Restaurant Segments (3D Clustering):** Unsupervised K-Means clustering groups restaurants based on Cost, Votes, and Ratings into distinct segments (e.g., Budget-Friendly, High-End Dining).
- **Top 10 Rated Restaurants:** A high-contrast horizontal bar chart showcasing the best-performing venues.
- **Service Breakdown:** Interactive pie charts displaying the distribution of Online Delivery and Table Booking availability.
- **Cost Distribution:** Box plots illustrating the spread of approximated costs across different restaurant types.

### 🤖 Machine Learning Integration
- **Rating Predictor Simulator:** Uses a trained `RandomForestRegressor` to predict a restaurant's rating in real-time. Adjust the sliders for Estimated Cost, Votes, and Services in the sidebar to simulate and predict the rating of a hypothetical restaurant!

### 🎛️ Dynamic Filtering
- **Interactive UI:** Built using `dash-bootstrap-components` with a sleek, dark analytical theme.
- **Real-Time KPIs:** Dynamic counters update instantly based on your filters to show Total Restaurants, Average Cost, Average Rating, and Total Votes.
- **Multi-Select Filters:** Filter the entire dashboard by Restaurant Type, Online Order availability, and Table Booking availability.

---

## 🛠️ Tech Stack

- **Python** 🐍
- **Dash & Dash Bootstrap Components** (Web Framework & UI)
- **Plotly Express & Graph Objects** (Interactive Visualizations)
- **Scikit-Learn** (Machine Learning: K-Means & Random Forest)
- **Pandas** (Data Manipulation & Cleaning)

---

## 🚀 Installation & Setup

To run this dashboard locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd <your-repository-folder>
   ```

2. **Install the required dependencies:**
   Make sure you have Python installed. Then run:
   ```bash
   pip install pandas plotly dash dash-bootstrap-components scikit-learn
   ```

3. **Run the Dashboard:**
   Ensure the dataset (`Zomato data .csv`) is in the same directory as the script.
   ```bash
   python "zomato eda analysis.py"
   ```

4. **View the App:**
   Open your web browser and navigate to the address shown in your terminal, usually:
   ```text
   http://127.0.0.1:8051/
   ```

---

## 📸 Screenshots
*(Add a screenshot of your beautiful dashboard running in the browser here! You can take a screenshot of the app running locally and replace this text with the image placeholder: `![Dashboard Screenshot](path/to/image.png)`)*

---

## 👨‍💻 Author

**G.V V KOWSHIK**
- [kowshik.gali](www.linkedin.com/in/kowshik-gali)
