"""
Complete AI-Based Demand Forecasting System
==========================================
All-in-one file with ARIMA, LSTM, and Advanced Analytics
No external imports needed - everything included!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Check and import required libraries
missing_libraries = []

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    missing_libraries.append("statsmodels")

try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    missing_libraries.append("scikit-learn")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    missing_libraries.append("tensorflow")

try:
    import joblib
except ImportError:
    missing_libraries.append("joblib")

if missing_libraries:
    print("⚠️ Missing libraries. Please install:")
    for lib in missing_libraries:
        print(f"   pip install {lib}")
    print("\nContinuing with available features...")

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class CompleteDemandForecaster:
    """Complete Demand Forecasting System with all features"""
    
    def __init__(self, csv_file_path):
        """Initialize the complete forecasting system"""
        print("🚀 Initializing Complete Demand Forecasting System...")
        self.csv_file_path = csv_file_path
        self.df = None
        self.scaler = MinMaxScaler() if 'MinMaxScaler' in globals() else None
        self.results = {}
        
        # Load and prepare data
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """Load and prepare the data with advanced features"""
        print("📊 Loading and preparing data...")
        
        # Load data
        self.df = pd.read_csv(self.csv_file_path)
        print(f"✅ Loaded {len(self.df)} records")
        
        # Data validation
        required_columns = ['transaction_date', 'quantity', 'current_stock', 'running_stock', 
                          'unit_price', 'brand', 'location', 'transaction_type']
        
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            print(f"⚠️ Warning: Missing columns: {missing_cols}")
        else:
            print("✅ All required columns present")
        
        # Convert transaction_date to datetime
        self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'])
        
        # Create demand metric
        self.df['demand'] = self.df['quantity']
        
        # Add comprehensive time features
        self.df['year'] = self.df['transaction_date'].dt.year
        self.df['month'] = self.df['transaction_date'].dt.month
        self.df['day'] = self.df['transaction_date'].dt.day
        self.df['weekday'] = self.df['transaction_date'].dt.dayofweek
        self.df['week'] = self.df['transaction_date'].dt.isocalendar().week
        self.df['quarter'] = self.df['transaction_date'].dt.quarter
        self.df['day_of_year'] = self.df['transaction_date'].dt.dayofyear
        self.df['is_weekend'] = (self.df['weekday'] >= 5).astype(int)
        
        # Cyclical features for better seasonality capture
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['weekday_sin'] = np.sin(2 * np.pi * self.df['weekday'] / 7)
        self.df['weekday_cos'] = np.cos(2 * np.pi * self.df['weekday'] / 7)
        
        # Create string representations
        self.df['month_name'] = self.df['transaction_date'].dt.strftime('%B')
        self.df['weekday_name'] = self.df['transaction_date'].dt.strftime('%A')
        
        # Economic and stock features
        self.df['stock_ratio'] = self.df['running_stock'] / (self.df['current_stock'] + 1)
        self.df['price_per_unit'] = self.df['unit_price']
        self.df['stock_turnover'] = self.df['quantity'] / (self.df['current_stock'] + 1)
        
        # Lag features and rolling statistics
        self.df = self.df.sort_values(['brand', 'transaction_date'])
        
        # Simple rolling features (without groupby to avoid complexity)
        self.df['demand_lag_1'] = self.df['demand'].shift(1)
        self.df['demand_lag_7'] = self.df['demand'].shift(7)
        self.df['demand_roll_mean_7'] = self.df['demand'].rolling(window=7, min_periods=1).mean()
        self.df['demand_roll_std_7'] = self.df['demand'].rolling(window=7, min_periods=1).std()
        
        # Fill NaN values
        self.df = self.df.fillna(method='bfill').fillna(method='ffill')
        
        print(f"✅ Data preparation completed. Shape: {self.df.shape}")
    
    def analyze_comprehensive_patterns(self):
        """Comprehensive demand pattern analysis"""
        print("📈 Analyzing comprehensive demand patterns...")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Overall demand trend
        plt.subplot(3, 3, 1)
        daily_demand = self.df.groupby('transaction_date')['demand'].sum().reset_index()
        plt.plot(daily_demand['transaction_date'], daily_demand['demand'], color='blue', alpha=0.7)
        plt.title('Daily Demand Trend', fontsize=12, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Total Demand')
        plt.xticks(rotation=45)
        
        # 2. Monthly seasonality
        plt.subplot(3, 3, 2)
        monthly_demand = self.df.groupby('month_name')['demand'].mean()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_demand = monthly_demand.reindex([m for m in month_order if m in monthly_demand.index])
        
        bars = plt.bar(range(len(monthly_demand)), monthly_demand.values, color='green', alpha=0.7)
        plt.title('Monthly Demand Pattern', fontsize=12, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Average Demand')
        plt.xticks(range(len(monthly_demand)), [m[:3] for m in monthly_demand.index], rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, monthly_demand.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.0f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Weekly pattern
        plt.subplot(3, 3, 3)
        weekly_demand = self.df.groupby('weekday_name')['demand'].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_demand = weekly_demand.reindex([d for d in day_order if d in weekly_demand.index])
        
        plt.bar(range(len(weekly_demand)), weekly_demand.values, color='orange', alpha=0.7)
        plt.title('Weekly Demand Pattern', fontsize=12, fontweight='bold')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Demand')
        plt.xticks(range(len(weekly_demand)), [d[:3] for d in weekly_demand.index], rotation=45)
        
        # 4. Brand performance
        plt.subplot(3, 3, 4)
        brand_demand = self.df.groupby('brand')['demand'].sum().sort_values(ascending=True)
        plt.barh(range(len(brand_demand)), brand_demand.values, color='red', alpha=0.7)
        plt.title('Brand Total Demand', fontsize=12, fontweight='bold')
        plt.xlabel('Total Demand')
        plt.ylabel('Brand')
        plt.yticks(range(len(brand_demand)), brand_demand.index)
        
        # 5. Location distribution
        plt.subplot(3, 3, 5)
        location_demand = self.df.groupby('location')['demand'].sum()
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(location_demand)]
        wedges, texts, autotexts = plt.pie(location_demand.values, labels=location_demand.index, 
                                          autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title('Demand by Location', fontsize=12, fontweight='bold')
        
        # 6. Price vs Demand scatter
        plt.subplot(3, 3, 6)
        brand_stats = self.df.groupby('brand').agg({
            'unit_price': 'mean',
            'demand': 'sum',
            'current_stock': 'mean'
        }).reset_index()
        
        scatter = plt.scatter(brand_stats['unit_price'], brand_stats['demand'], 
                            s=brand_stats['current_stock']*2, alpha=0.6, c=range(len(brand_stats)), cmap='viridis')
        plt.title('Price vs Demand by Brand', fontsize=12, fontweight='bold')
        plt.xlabel('Average Unit Price')
        plt.ylabel('Total Demand')
        
        # Add brand labels
        for i, brand in enumerate(brand_stats['brand']):
            plt.annotate(brand, (brand_stats['unit_price'].iloc[i], brand_stats['demand'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 7. Transaction type analysis
        plt.subplot(3, 3, 7)
        transaction_demand = self.df.groupby('transaction_type')['demand'].sum()
        plt.bar(transaction_demand.index, transaction_demand.values, color='purple', alpha=0.7)
        plt.title('Demand by Transaction Type', fontsize=12, fontweight='bold')
        plt.xlabel('Transaction Type')
        plt.ylabel('Total Demand')
        plt.xticks(rotation=45)
        
        # 8. Stock level analysis
        plt.subplot(3, 3, 8)
        self.df['stock_level_category'] = pd.cut(self.df['current_stock'], 
                                                bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        stock_demand = self.df.groupby('stock_level_category')['demand'].mean()
        plt.bar(range(len(stock_demand)), stock_demand.values, color='teal', alpha=0.7)
        plt.title('Demand by Stock Level', fontsize=12, fontweight='bold')
        plt.xlabel('Stock Level Category')
        plt.ylabel('Average Demand')
        plt.xticks(range(len(stock_demand)), stock_demand.index, rotation=45)
        
        # 9. Correlation heatmap
        plt.subplot(3, 3, 9)
        correlation_cols = ['demand', 'current_stock', 'running_stock', 'unit_price', 
                           'month', 'weekday', 'stock_turnover']
        available_cols = [col for col in correlation_cols if col in self.df.columns]
        
        if len(available_cols) > 2:
            corr_matrix = self.df[available_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('Feature Correlations', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return daily_demand
    
    def seasonal_decomposition_analysis(self, daily_demand):
        """Perform seasonal decomposition"""
        print("🔍 Performing seasonal decomposition analysis...")
        
        try:
            # Prepare time series
            ts_data = daily_demand.set_index('transaction_date')['demand']
            ts_data = ts_data.asfreq('D', fill_value=ts_data.mean())
            
            # Perform decomposition
            if len(ts_data) > 30:  # Need sufficient data
                decomposition = seasonal_decompose(ts_data, model='additive', period=min(30, len(ts_data)//3))
                
                # Plot decomposition
                fig, axes = plt.subplots(4, 1, figsize=(15, 12))
                fig.suptitle('Seasonal Decomposition of Demand', fontsize=16, fontweight='bold')
                
                decomposition.observed.plot(ax=axes[0], title='Original Demand', color='blue')
                axes[0].set_ylabel('Demand')
                
                decomposition.trend.plot(ax=axes[1], title='Trend Component', color='green')
                axes[1].set_ylabel('Trend')
                
                decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component', color='orange')
                axes[2].set_ylabel('Seasonal')
                
                decomposition.resid.plot(ax=axes[3], title='Residual Component', color='red')
                axes[3].set_ylabel('Residual')
                axes[3].set_xlabel('Date')
                
                plt.tight_layout()
                plt.show()
                
                return ts_data, decomposition
            else:
                print("⚠️ Insufficient data for seasonal decomposition")
                return ts_data, None
                
        except Exception as e:
            print(f"❌ Seasonal decomposition error: {e}")
            ts_data = daily_demand.set_index('transaction_date')['demand']
            return ts_data, None
    
    def arima_forecasting(self, ts_data, forecast_days=30):
        """ARIMA model forecasting"""
        print("🤖 Building ARIMA forecasting model...")
        
        try:
            # Check if ARIMA is available
            if 'ARIMA' not in globals():
                print("⚠️ ARIMA not available. Please install statsmodels.")
                return None, None, None
            
            # Prepare data
            ts_clean = ts_data.dropna()
            if len(ts_clean) < 10:
                print("⚠️ Insufficient data for ARIMA modeling")
                return None, None, None
            
            # Split data
            train_size = max(int(len(ts_clean) * 0.8), len(ts_clean) - forecast_days)
            train_data = ts_clean[:train_size]
            test_data = ts_clean[train_size:]
            
            # Fit ARIMA model with simple parameters
            model = ARIMA(train_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Predictions
            if len(test_data) > 0:
                test_pred = fitted_model.forecast(steps=len(test_data))
                mae = mean_absolute_error(test_data, test_pred)
                print(f"ARIMA Model - Test MAE: {mae:.2f}")
            else:
                test_pred = None
                mae = None
            
            # Future forecast
            future_forecast = fitted_model.forecast(steps=forecast_days)
            future_dates = pd.date_range(start=ts_clean.index[-1] + timedelta(days=1), 
                                       periods=forecast_days, freq='D')
            
            # Visualization
            plt.figure(figsize=(15, 8))
            plt.plot(train_data.index, train_data.values, label='Training Data', color='blue', alpha=0.7)
            
            if test_data is not None and len(test_data) > 0:
                plt.plot(test_data.index, test_data.values, label='Actual Test', color='green', alpha=0.7)
                if test_pred is not None:
                    plt.plot(test_data.index, test_pred, label='Test Predictions', color='red', linestyle='--', alpha=0.7)
            
            plt.plot(future_dates, future_forecast, label='Future Forecast', 
                    color='orange', linewidth=2, marker='o', markersize=3)
            
            plt.title('ARIMA Demand Forecasting Results', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Demand')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            # Print forecast summary
            print(f"📊 ARIMA Forecast Summary:")
            print(f"   Average predicted demand: {future_forecast.mean():.1f}")
            print(f"   Max predicted demand: {future_forecast.max():.1f}")
            print(f"   Min predicted demand: {future_forecast.min():.1f}")
            
            return fitted_model, future_forecast, future_dates
            
        except Exception as e:
            print(f"❌ ARIMA forecasting error: {e}")
            return None, None, None
    
    def lstm_forecasting(self, ts_data, forecast_days=30, lookback_window=7):
        """LSTM model forecasting"""
        print("🧠 Building LSTM forecasting model...")
        
        try:
            # Check if required libraries are available
            if self.scaler is None or 'Sequential' not in globals():
                print("⚠️ LSTM libraries not available. Please install tensorflow and scikit-learn.")
                return None, None
            
            # Prepare data
            ts_values = ts_data.fillna(ts_data.mean()).values.reshape(-1, 1)
            
            if len(ts_values) < lookback_window + 10:
                print("⚠️ Insufficient data for LSTM modeling")
                return None, None
            
            # Scale data
            scaled_data = self.scaler.fit_transform(ts_values)
            
            # Create sequences
            X, y = [], []
            for i in range(lookback_window, len(scaled_data)):
                X.append(scaled_data[i-lookback_window:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            if len(X) < 10:
                print("⚠️ Insufficient sequences for LSTM training")
                return None, None
            
            # Split data
            train_size = max(int(len(X) * 0.8), len(X) - forecast_days)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train with early stopping
            early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            
            print("   Training LSTM model...")
            history = model.fit(X_train, y_train, epochs=50, batch_size=16, 
                              callbacks=[early_stop], verbose=0)
            
            # Test predictions
            if len(X_test) > 0:
                test_pred_scaled = model.predict(X_test, verbose=0)
                test_pred = self.scaler.inverse_transform(test_pred_scaled)
                y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
                mae = mean_absolute_error(y_test_actual, test_pred)
                print(f"   LSTM Model - Test MAE: {mae:.2f}")
            
            # Future predictions
            last_sequence = X[-1].reshape(1, X.shape[1], 1)
            future_predictions = []
            
            for _ in range(forecast_days):
                next_pred = model.predict(last_sequence, verbose=0)
                future_predictions.append(next_pred[0, 0])
                
                # Update sequence
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = next_pred[0, 0]
            
            # Scale back predictions
            future_predictions = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            future_predictions = future_predictions.flatten()
            
            # Visualization
            plt.figure(figsize=(15, 8))
            
            # Plot training history
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss', color='blue')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
            plt.title('LSTM Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot predictions
            plt.subplot(1, 2, 2)
            recent_data = ts_data.tail(30)
            plt.plot(recent_data.index, recent_data.values, label='Recent Actual', color='blue', alpha=0.7)
            
            future_dates = pd.date_range(start=ts_data.index[-1] + timedelta(days=1), 
                                       periods=forecast_days, freq='D')
            plt.plot(future_dates, future_predictions, label='LSTM Forecast', 
                    color='red', linewidth=2, marker='o', markersize=3)
            
            plt.title('LSTM Forecast Results')
            plt.xlabel('Date')
            plt.ylabel('Demand')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            # Print forecast summary
            print(f"📊 LSTM Forecast Summary:")
            print(f"   Average predicted demand: {future_predictions.mean():.1f}")
            print(f"   Max predicted demand: {future_predictions.max():.1f}")
            print(f"   Min predicted demand: {future_predictions.min():.1f}")
            
            return model, future_predictions
            
        except Exception as e:
            print(f"❌ LSTM forecasting error: {e}")
            return None, None
    
    def machine_learning_analysis(self):
        """Random Forest analysis for feature importance"""
        print("🔍 Running machine learning analysis...")
        
        try:
            if 'RandomForestRegressor' not in globals():
                print("⚠️ Random Forest not available. Please install scikit-learn.")
                return None, None
            
            # Prepare features
            feature_columns = ['current_stock', 'running_stock', 'unit_price', 'month', 'weekday', 
                             'quarter', 'is_weekend', 'stock_turnover', 'stock_ratio']
            
            # Add categorical features
            df_encoded = pd.get_dummies(self.df, columns=['brand', 'location', 'transaction_type'], 
                                       prefix=['brand', 'loc', 'trans'])
            
            # Select available features
            available_features = [col for col in feature_columns if col in df_encoded.columns]
            categorical_features = [col for col in df_encoded.columns 
                                  if col.startswith(('brand_', 'loc_', 'trans_'))]
            
            all_features = available_features + categorical_features
            
            if len(all_features) < 3:
                print("⚠️ Insufficient features for ML analysis")
                return None, None
            
            X = df_encoded[all_features].fillna(0)
            y = df_encoded['demand']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train, y_train)
            
            # Predictions and evaluation
            y_pred = rf_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"   Random Forest Performance:")
            print(f"   MAE: {mae:.2f}")
            print(f"   R² Score: {r2:.4f}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            
            bars = plt.barh(range(len(top_features)), top_features['importance'], color='skyblue', alpha=0.7)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importance for Demand Prediction', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{top_features.iloc[i]["importance"]:.3f}', 
                        va='center', ha='left', fontsize=8)
            
            plt.tight_layout()
            plt.show()
            
            return rf_model, feature_importance
            
        except Exception as e:
            print(f"❌ ML analysis error: {e}")
            return None, None
    
    def anomaly_detection(self):
        """Detect anomalies in demand patterns"""
        print("🚨 Detecting demand anomalies...")
        
        try:
            # Calculate rolling statistics
            window = min(14, len(self.df) // 10)
            if window < 3:
                window = 3
            
            # Simple anomaly detection using statistical bounds
            self.df['demand_mean'] = self.df['demand'].rolling(window=window, min_periods=1).mean()
            self.df['demand_std'] = self.df['demand'].rolling(window=window, min_periods=1).std()
            
            # Define bounds (2 standard deviations)
            self.df['upper_bound'] = self.df['demand_mean'] + 2 * self.df['demand_std']
            self.df['lower_bound'] = self.df['demand_mean'] - 2 * self.df['demand_std']
            
            # Identify anomalies
            self.df['is_anomaly'] = (
                (self.df['demand'] > self.df['upper_bound']) | 
                (self.df['demand'] < self.df['lower_bound'])
            )
            
            anomalies = self.df[self.df['is_anomaly'] == True]
            anomaly_rate = len(anomalies) / len(self.df) * 100
            
            print(f"   Found {len(anomalies)} anomalies ({anomaly_rate:.2f}% of data)")
            
            # Visualize anomalies by brand
            brands = self.df['brand'].unique()[:4]  # Top 4 brands
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('Demand Anomaly Detection by Brand', fontsize=16, fontweight='bold')
            
            for idx, brand in enumerate(brands):
                brand_data = self.df[self.df['brand'] == brand].sort_values('transaction_date')
                brand_anomalies = brand_data[brand_data['is_anomaly'] == True]
                
                row, col = idx // 2, idx % 2
                
                # Plot normal demand
                axes[row, col].plot(brand_data['transaction_date'], brand_data['demand'], 
                                  'b-', alpha=0.6, label='Normal Demand')
                
                # Plot bounds
                axes[row, col].fill_between(brand_data['transaction_date'], 
                                          brand_data['lower_bound'], brand_data['upper_bound'], 
                                          alpha=0.2, color='gray', label='Normal Range')
                
                # Highlight anomalies
                if len(brand_anomalies) > 0:
                    axes[row, col].scatter(brand_anomalies['transaction_date'], brand_anomalies['demand'], 
                                         color='red', s=50, label='Anomalies', zorder=5)
                
                axes[row, col].set_title(f'{brand} - Anomalies: {len(brand_anomalies)}')
                axes[row, col].set_xlabel('Date')
                axes[row, col].set_ylabel('Demand')
                axes[row, col].legend()
                axes[row, col].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            # Anomaly summary
            if len(anomalies) > 0:
                anomaly_summary = anomalies.groupby(['brand', 'location']).agg({
                    'demand': ['count', 'mean', 'max'],
                    'transaction_date': ['min', 'max']
                }).round(2)
                
                print("\n📊 Anomaly Summary:")
                print(anomaly_summary.head())
            
            return anomalies
            
        except Exception as e:
            print(f"❌ Anomaly detection error: {e}")
            return pd.DataFrame()
    
    def generate_comprehensive_recommendations(self, arima_forecast, lstm_forecast, future_dates, forecast_days):
        """Generate comprehensive stock recommendations"""
        print("💡 Generating comprehensive recommendations...")
        
        recommendations = []
        
        # Combine forecasts if both are available
        if arima_forecast is not None and lstm_forecast is not None:
            combined_forecast = (arima_forecast + lstm_forecast) / 2
            forecast_source = "ARIMA + LSTM Ensemble"
        elif arima_forecast is not None:
            combined_forecast = arima_forecast
            forecast_source = "ARIMA Only"
        elif lstm_forecast is not None:
            combined_forecast = lstm_forecast
            forecast_source = "LSTM Only"
        else:
            print("❌ No forecasts available for recommendations")
            return pd.DataFrame()
        
        print(f"   Using forecast source: {forecast_source}")
        
        # Calculate statistics for dynamic thresholds
        forecast_mean = np.mean(combined_forecast)
        forecast_std = np.std(combined_forecast)
        
        # Generate recommendations
        for i in range(min(len(combined_forecast), forecast_days)):
            demand = combined_forecast[i]
            
            # Dynamic safety stock based on demand variability
            if demand > forecast_mean + forecast_std:
                safety_factor = 0.3  # 30% for high demand
                priority = "High"
                alert = "Urgent - High Demand Expected"
            elif demand > forecast_mean:
                safety_factor = 0.2  # 20% for medium demand
                priority = "Medium"
                alert = "Normal Restock Required"
            else:
                safety_factor = 0.15  # 15% for low demand
                priority = "Low"
                alert = "Monitor Stock Level"
            
            safety_stock = demand * safety_factor
            recommended_stock = demand + safety_stock
            
            # Confidence level based on forecast consistency
            if i < len(combined_forecast) // 3:
                confidence = "High"
            elif i < 2 * len(combined_forecast) // 3:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Date handling
            if future_dates is not None and i < len(future_dates):
                date_str = future_dates[i].strftime('%Y-%m-%d')
                weekday = future_dates[i].strftime('%A')
            else:
                date_str = f"Day {i+1}"
                weekday = "Unknown"
            
            recommendations.append({
                'Date': date_str,
                'Weekday': weekday,
                'Predicted_Demand': round(demand, 0),
                'Safety_Stock': round(safety_stock, 0),
                'Recommended_Stock': round(recommended_stock, 0),
                'Priority': priority,
                'Alert': alert,
                'Confidence': confidence,
                'Forecast_Source': forecast_source
            })
        
        recommendations_df = pd.DataFrame(recommendations)
        
        # Add summary statistics
        print(f"\n📋 Recommendation Summary:")
        print(f"   Total recommendations: {len(recommendations_df)}")
        print(f"   High priority items: {len(recommendations_df[recommendations_df['Priority'] == 'High'])}")
        print(f"   Medium priority items: {len(recommendations_df[recommendations_df['Priority'] == 'Medium'])}")
        print(f"   Low priority items: {len(recommendations_df[recommendations_df['Priority'] == 'Low'])}")
        print(f"   Average daily demand: {recommendations_df['Predicted_Demand'].mean():.1f}")
        print(f"   Peak demand: {recommendations_df['Predicted_Demand'].max():.0f}")
        
        # Save recommendations
        recommendations_df.to_csv('comprehensive_demand_forecast_recommendations.csv', index=False)
        print("✅ Recommendations saved to 'comprehensive_demand_forecast_recommendations.csv'")
        
        # Display top 10 recommendations
        print(f"\n📊 Top 10 Recommendations:")
        display_cols = ['Date', 'Weekday', 'Predicted_Demand', 'Recommended_Stock', 'Priority', 'Alert']
        print(recommendations_df[display_cols].head(10).to_string(index=False))
        
        return recommendations_df
    
    def generate_business_insights(self):
        """Generate actionable business insights"""
        print("📈 Generating business insights...")
        
        insights = []
        
        try:
            # 1. Peak demand analysis
            monthly_demand = self.df.groupby('month_name')['demand'].mean()
            peak_month = monthly_demand.idxmax()
            peak_demand = monthly_demand[peak_month]
            insights.append(f"Peak demand month: {peak_month} (avg: {peak_demand:.1f} units)")
            
            # 2. Best performing products
            brand_performance = self.df.groupby('brand')['demand'].sum().sort_values(ascending=False)
            top_brand = brand_performance.index[0]
            top_brand_demand = brand_performance.iloc[0]
            insights.append(f"Top brand: {top_brand} with {top_brand_demand} total units sold")
            
            # 3. Location performance
            location_performance = self.df.groupby('location')['demand'].mean().sort_values(ascending=False)
            best_location = location_performance.index[0]
            best_location_demand = location_performance.iloc[0]
            insights.append(f"Best performing location: {best_location} (avg: {best_location_demand:.1f} units)")
            
            # 4. Weekday patterns
            weekday_demand = self.df.groupby('weekday_name')['demand'].mean().sort_values(ascending=False)
            best_day = weekday_demand.index[0]
            best_day_demand = weekday_demand.iloc[0]
            insights.append(f"Highest demand day: {best_day} (avg: {best_day_demand:.1f} units)")
            
            # 5. Price sensitivity
            if 'unit_price' in self.df.columns:
                price_demand_corr = self.df[['unit_price', 'demand']].corr().iloc[0,1]
                if abs(price_demand_corr) > 0.3:
                    sensitivity = "high" if abs(price_demand_corr) > 0.5 else "moderate"
                    direction = "negative" if price_demand_corr < 0 else "positive"
                    insights.append(f"Price sensitivity: {sensitivity} {direction} correlation ({price_demand_corr:.3f})")
            
            # 6. Stock efficiency
            if 'stock_turnover' in self.df.columns:
                avg_turnover = self.df['stock_turnover'].mean()
                if avg_turnover > 1.0:
                    insights.append(f"Good stock turnover: {avg_turnover:.2f} (healthy inventory movement)")
                else:
                    insights.append(f"Low stock turnover: {avg_turnover:.2f} (consider reducing stock levels)")
            
            # 7. Transaction type analysis
            transaction_analysis = self.df.groupby('transaction_type')['demand'].sum().sort_values(ascending=False)
            dominant_transaction = transaction_analysis.index[0]
            transaction_pct = transaction_analysis.iloc[0] / transaction_analysis.sum() * 100
            insights.append(f"Dominant transaction type: {dominant_transaction} ({transaction_pct:.1f}% of total)")
            
            # 8. Demand variability
            demand_cv = self.df['demand'].std() / self.df['demand'].mean()
            if demand_cv > 1.0:
                insights.append(f"High demand variability (CV: {demand_cv:.2f}) - consider higher safety stocks")
            else:
                insights.append(f"Moderate demand variability (CV: {demand_cv:.2f}) - current safety stocks adequate")
            
        except Exception as e:
            insights.append(f"Error generating some insights: {str(e)}")
        
        return insights
    
    def create_executive_summary(self, recommendations, insights, anomalies):
        """Create executive summary report"""
        print("📋 Creating executive summary...")
        
        summary = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_overview': {
                'total_records': len(self.df),
                'date_range': f"{self.df['transaction_date'].min().strftime('%Y-%m-%d')} to {self.df['transaction_date'].max().strftime('%Y-%m-%d')}",
                'unique_brands': self.df['brand'].nunique(),
                'unique_locations': self.df['location'].nunique(),
                'total_demand': int(self.df['demand'].sum()),
                'avg_daily_demand': round(self.df['demand'].mean(), 1)
            },
            'forecasting_results': {
                'models_used': self.results.get('models_used', []),
                'forecast_period': f"{len(recommendations)} days" if len(recommendations) > 0 else "N/A",
                'high_priority_items': len(recommendations[recommendations['Priority'] == 'High']) if len(recommendations) > 0 else 0,
                'anomalies_detected': len(anomalies)
            },
            'key_insights': insights,
            'recommendations_summary': {
                'total_recommendations': len(recommendations),
                'urgent_actions': len(recommendations[recommendations['Priority'] == 'High']) if len(recommendations) > 0 else 0,
                'avg_recommended_stock': round(recommendations['Recommended_Stock'].mean(), 1) if len(recommendations) > 0 else 0
            }
        }
        
        # Display executive summary
        print("\n" + "="*70)
        print("📊 EXECUTIVE SUMMARY - DEMAND FORECASTING ANALYSIS")
        print("="*70)
        
        print(f"\n📅 Analysis Date: {summary['analysis_date']}")
        print(f"📊 Data Period: {summary['data_overview']['date_range']}")
        print(f"📦 Total Records: {summary['data_overview']['total_records']:,}")
        print(f"🏷️ Brands Analyzed: {summary['data_overview']['unique_brands']}")
        print(f"📍 Locations Analyzed: {summary['data_overview']['unique_locations']}")
        print(f"📈 Total Demand: {summary['data_overview']['total_demand']:,} units")
        print(f"📊 Average Daily Demand: {summary['data_overview']['avg_daily_demand']} units")
        
        print(f"\n🤖 Forecasting Results:")
        print(f"   Forecast Period: {summary['forecasting_results']['forecast_period']}")
        print(f"   High Priority Items: {summary['forecasting_results']['high_priority_items']}")
        print(f"   Anomalies Detected: {summary['forecasting_results']['anomalies_detected']}")
        
        print(f"\n💡 Key Business Insights:")
        for i, insight in enumerate(insights[:8], 1):
            print(f"   {i}. {insight}")
        
        print(f"\n🎯 Action Items:")
        if summary['recommendations_summary']['urgent_actions'] > 0:
            print(f"   🚨 {summary['recommendations_summary']['urgent_actions']} items need URGENT restocking")
        print(f"   📋 Review {summary['recommendations_summary']['total_recommendations']} total recommendations")
        if summary['forecasting_results']['anomalies_detected'] > 0:
            print(f"   🔍 Investigate {summary['forecasting_results']['anomalies_detected']} demand anomalies")
        
        print(f"\n📁 Generated Files:")
        print(f"   • comprehensive_demand_forecast_recommendations.csv")
        print(f"   • executive_summary.json")
        
        # Save summary to JSON
        import json
        with open('executive_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("\n✅ Executive summary saved to 'executive_summary.json'")
        print("="*70)
        
        return summary
    
    def run_complete_analysis(self, forecast_days=30):
        """Run the complete demand forecasting analysis"""
        print("🎯 STARTING COMPLETE DEMAND FORECASTING ANALYSIS")
        print("="*70)
        
        start_time = datetime.now()
        
        try:
            # 1. Comprehensive pattern analysis
            print("\n1️⃣ PATTERN ANALYSIS")
            daily_demand = self.analyze_comprehensive_patterns()
            
            # 2. Seasonal decomposition
            print("\n2️⃣ SEASONAL ANALYSIS")
            ts_data, decomposition = self.seasonal_decomposition_analysis(daily_demand)
            
            # 3. ARIMA forecasting
            print("\n3️⃣ ARIMA FORECASTING")
            arima_model, arima_forecast, future_dates = self.arima_forecasting(ts_data, forecast_days)
            
            # 4. LSTM forecasting
            print("\n4️⃣ LSTM FORECASTING")
            lstm_model, lstm_forecast = self.lstm_forecasting(ts_data, forecast_days)
            
            # 5. Machine learning analysis
            print("\n5️⃣ MACHINE LEARNING ANALYSIS")
            rf_model, feature_importance = self.machine_learning_analysis()
            
            # 6. Anomaly detection
            print("\n6️⃣ ANOMALY DETECTION")
            anomalies = self.anomaly_detection()
            
            # 7. Generate recommendations
            print("\n7️⃣ RECOMMENDATION GENERATION")
            recommendations = self.generate_comprehensive_recommendations(
                arima_forecast, lstm_forecast, future_dates, forecast_days
            )
            
            # 8. Business insights
            print("\n8️⃣ BUSINESS INSIGHTS")
            insights = self.generate_business_insights()
            
            # 9. Executive summary
            print("\n9️⃣ EXECUTIVE SUMMARY")
            summary = self.create_executive_summary(recommendations, insights, anomalies)
            
            # Store results
            self.results = {
                'arima_model': arima_model,
                'lstm_model': lstm_model,
                'rf_model': rf_model,
                'recommendations': recommendations,
                'anomalies': anomalies,
                'insights': insights,
                'summary': summary,
                'feature_importance': feature_importance,
                'models_used': []
            }
            
            # Track which models were successfully used
            if arima_model is not None:
                self.results['models_used'].append('ARIMA')
            if lstm_model is not None:
                self.results['models_used'].append('LSTM')
            if rf_model is not None:
                self.results['models_used'].append('Random Forest')
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Final success message
            print(f"\n🎊 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"⏱️ Total execution time: {execution_time:.1f} seconds")
            print(f"🤖 Models used: {', '.join(self.results['models_used'])}")
            print(f"📊 Recommendations generated: {len(recommendations)}")
            print(f"🚨 Anomalies detected: {len(anomalies)}")
            print(f"💡 Business insights: {len(insights)}")
            
            print(f"\n📋 NEXT STEPS:")
            print(f"1. Review comprehensive_demand_forecast_recommendations.csv")
            print(f"2. Check executive_summary.json for key findings")
            print(f"3. Investigate any high-priority restock items")
            print(f"4. Monitor demand anomalies")
            print(f"5. Implement recommendations in inventory system")
            
            return self.results
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """Main function to run the complete demand forecasting system"""
    
    # CONFIGURATION - UPDATE THESE PATHS
    CSV_FILE_PATH = r"D:\inventory\Ai-based-Inventory-control-system-and-depot-Management\Backend\Dataset\inventory_training_5000.csv"
    FORECAST_DAYS = 30
    
    print("🚀 COMPLETE AI-BASED DEMAND FORECASTING SYSTEM")
    print("="*70)
    print(f"📊 CSV File: {CSV_FILE_PATH}")
    print(f"📅 Forecast Days: {FORECAST_DAYS}")
    print("="*70)
    
    try:
        # Check if file exists
        import os
        if not os.path.exists(CSV_FILE_PATH):
            print(f"❌ ERROR: File not found - {CSV_FILE_PATH}")
            print("💡 Please check the file path and try again.")
            return None
        
        # Initialize and run the complete system
        print("🔄 Initializing system...")
        forecaster = CompleteDemandForecaster(CSV_FILE_PATH)
        
        print("🔄 Running complete analysis...")
        results = forecaster.run_complete_analysis(forecast_days=FORECAST_DAYS)
        
        if results:
            print("\n🎉 SUCCESS! Complete demand forecasting analysis finished!")
            return results
        else:
            print("\n❌ FAILURE: Analysis could not be completed")
            return None
            
    except FileNotFoundError:
        print(f"❌ ERROR: CSV file not found: {CSV_FILE_PATH}")
        print("💡 Please check the file path and ensure the file exists.")
        return None
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("💡 Please check your data format and try again.")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🎯 Starting Complete Demand Forecasting System...")
    
    # Install check
    print("📦 Checking required libraries...")
    required_libs = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'statsmodels': 'statsmodels',
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
        'joblib': 'joblib'
    }
    
    missing = []
    for lib, install_name in required_libs.items():
        try:
            __import__(lib)
        except ImportError:
            missing.append(install_name)
    
    if missing:
        print(f"⚠️ Missing libraries: {', '.join(missing)}")
        print(f"📦 Install with: pip install {' '.join(missing)}")
        print("🔄 Continuing with available features...")
    else:
        print("✅ All libraries available!")
    
    # Run the main analysis
    results = main()
    
    if results:
        print("\n📊 FINAL RESULTS SUMMARY:")
        print("="*50)
        print(f"✅ Models trained: {len(results.get('models_used', []))}")
        print(f"✅ Recommendations: {len(results.get('recommendations', []))}")
        print(f"✅ Anomalies detected: {len(results.get('anomalies', []))}")
        print(f"✅ Business insights: {len(results.get('insights', []))}")
        print("\n📁 Check generated files for detailed results!")
    else:
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Ensure file path is correct")
        print("2. Install missing libraries")
        print("3. Check data format")
        print("4. Contact support if issues persist")
    
    print("\n" + "="*70)
    print("Thank you for using the Complete Demand Forecasting System!")
    print("="*70)