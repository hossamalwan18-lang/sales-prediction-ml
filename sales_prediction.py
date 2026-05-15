import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


class SalesPredictor:
    def __init__(self, url):
        self.url = url
        self.df = None

        # Pipeline احترافي
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])

    def load_and_clean_data(self):
        """تحميل وتنظيف البيانات"""

        print("🔍 [1/6] Loading Dataset...")

        # تحميل البيانات
        self.df = pd.read_csv(self.url, index_col=0)

        # حذف القيم المفقودة
        if self.df.isnull().sum().sum() > 0:
            self.df.dropna(inplace=True)

        print(f"✅ Dataset Loaded Successfully!")
        print(f"📦 Records: {self.df.shape[0]}")
        print(f"📊 Features: {self.df.shape[1]}")

    def exploratory_analysis(self):
        """تحليل البيانات"""

        print("📊 [2/6] Performing Exploratory Data Analysis...")

        # Correlation Matrix
        corr = self.df.corr()

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            corr,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.5
        )

        plt.title(
            'Correlation Matrix - Advertising Dataset',
            fontsize=14,
            fontweight='bold'
        )

        plt.savefig(
            'correlation_heatmap.png',
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print("✅ Heatmap Saved Successfully!")

    def train_model(self):
        """تدريب الموديل"""

        print("🤖 [3/6] Training Machine Learning Model...")

        # Features
        X = self.df[['TV', 'radio', 'newspaper']]

        # Target
        y = self.df['sales']

        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # تدريب Pipeline
        self.pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = self.pipeline.predict(X_test)

        # Cross Validation
        cv_scores = cross_val_score(
            self.pipeline,
            X,
            y,
            cv=5,
            scoring='r2'
        )

        # مقارنة مع Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        rf_scores = cross_val_score(
            rf_model,
            X,
            y,
            cv=5,
            scoring='r2'
        )

        # Metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'cv_mean': cv_scores.mean(),
            'rf_mean': rf_scores.mean()
        }

        # حفظ النظام بالكامل
        joblib.dump(self.pipeline, 'sales_prediction_system.pkl')

        # عرض توقعات
        results = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred
        })

        print("\n📌 Sample Predictions:")
        print(results.head())

        return X_test, y_test, y_pred, metrics

    def display_metrics(self, metrics):
        """عرض النتائج"""

        print("\n" + "=" * 50)
        print("🏆 MODEL PERFORMANCE")
        print("=" * 50)

        print(f"📈 R² Score           : {metrics['r2']:.4f}")
        print(f"📉 Mean Squared Error : {metrics['mse']:.4f}")
        print(f"📊 Mean Absolute Error: {metrics['mae']:.4f}")
        print(f"🔄 Cross Validation   : {metrics['cv_mean']:.4f}")
        print(f"🌳 Random Forest Score: {metrics['rf_mean']:.4f}")

        print("=" * 50)

    def plot_results(self, X_test, y_test, y_pred):
        """رسم النتائج"""

        print("🎨 [4/6] Creating Visualizations...")

        # استخراج coefficients من Linear Regression
        model = self.pipeline.named_steps['model']

        features = ['TV', 'Radio', 'Newspaper']
        coefficients = model.coef_

        plt.figure(figsize=(14, 6))

        # الرسم الأول
        plt.subplot(1, 2, 1)

        bars = plt.bar(features, coefficients)

        plt.title('Feature Importance')
        plt.xlabel('Advertising Channels')
        plt.ylabel('Coefficient Value')

        # كتابة القيم فوق الأعمدة
        for bar in bars:
            yval = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                round(yval, 2),
                ha='center',
                va='bottom'
            )

        # الرسم الثاني
        plt.subplot(1, 2, 2)

        plt.scatter(
            y_test,
            y_pred,
            alpha=0.7,
            edgecolors='black'
        )

        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--',
            linewidth=2
        )

        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.title('Actual vs Predicted Sales')

        plt.tight_layout()

        plt.savefig(
            'model_results.png',
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print("✅ Visualizations Saved Successfully!")

    def residual_analysis(self, y_test, y_pred):
        """تحليل الأخطاء"""

        print("📉 [5/6] Performing Residual Analysis...")

        residuals = y_test - y_pred

        plt.figure(figsize=(8, 6))

        plt.scatter(y_pred, residuals, alpha=0.7)

        plt.axhline(y=0, linestyle='--')

        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Analysis')

        plt.savefig(
            'residual_plot.png',
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print("✅ Residual Plot Saved Successfully!")

    def predict_new(self, tv, radio, newspaper):
        """التنبؤ ببيانات جديدة"""

        saved_pipeline = joblib.load('sales_prediction_system.pkl')

        new_data = pd.DataFrame({
            'TV': [tv],
            'radio': [radio],
            'newspaper': [newspaper]
        })

        prediction = saved_pipeline.predict(new_data)[0]

        return prediction


if __name__ == '__main__':

    # Dataset URL
    DATA_URL = 'https://www.statlearning.com/s/Advertising.csv'

    # إنشاء الكلاس
    predictor = SalesPredictor(DATA_URL)

    # تشغيل المشروع
    predictor.load_and_clean_data()

    predictor.exploratory_analysis()

    X_test, y_test, y_pred, metrics = predictor.train_model()

    predictor.display_metrics(metrics)

    predictor.plot_results(X_test, y_test, y_pred)

    predictor.residual_analysis(y_test, y_pred)

    # تجربة تنبؤ جديدة
    print("\n✨ [6/6] New Prediction Example")

    prediction = predictor.predict_new(
        tv=200,
        radio=40,
        newspaper=10
    )

    print(f"💰 Expected Sales: {prediction:.2f} thousand units")

    print("\n📁 Files Generated:")
    print("1. correlation_heatmap.png")
    print("2. model_results.png")
    print("3. residual_plot.png")
    print("4. sales_prediction_system.pkl")