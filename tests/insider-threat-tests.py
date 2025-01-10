# Create a directory for test files
import os
if not os.path.exists('test_cases'):
    os.makedirs('test_cases')

# Generate test files with various scenarios
test_cases = {
    'normal_behavior.csv': pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=5, freq='H'),
        'from': ['employee@company.com'] * 5,
        'to': ['colleague1@company.com; colleague2@company.com'] * 5,
        'cc': ['manager@company.com'] * 5,
        'bcc': [''] * 5,
        'size': [1024] * 5,  # Normal email size
        'attachments': [1] * 5,  # Normal attachment count
        'content': [
            'Sending the quarterly report for review.',
            'Meeting notes from today\'s discussion.',
            'Updated project timeline attached.',
            'Weekly status update for Team Alpha.',
            'Resource allocation for next sprint.'
        ]
    }),
    
    'suspicious_behavior.csv': pd.DataFrame({
        'date': pd.date_range(start='2024-01-01 23:00:00', periods=5, freq='H'),
        'from': ['employee@company.com'] * 5,
        'to': ['external@competitor.com; personal@gmail.com'] * 5,
        'cc': [''] * 5,
        'bcc': ['hidden@external.com'] * 5,
        'size': [50000] * 5,  # Large email size
        'attachments': [5] * 5,  # Many attachments
        'content': [
            'Database backup files attached.',
            'Complete customer list export.',
            'Source code repository backup.',
            'Financial forecasts for all departments.',
            'Employee personal information records.'
        ]
    }),
    
    'data_exfiltration.csv': pd.DataFrame({
        'date': pd.date_range(start='2024-01-01 02:00:00', periods=5, freq='H'),
        'from': ['employee@company.com'] * 5,
        'to': ['personal@protonmail.com'] * 5,
        'cc': [''] * 5,
        'bcc': [''] * 5,
        'size': [100000] * 5,  # Very large email size
        'attachments': [3] * 5,
        'content': [
            'ZIP archive of project files',
            'Encrypted database backup',
            'Complete source code archive',
            'Full customer database export',
            'Confidential financial reports'
        ]
    }),
    
    'mixed_behavior.csv': pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=5, freq='D'),
        'from': ['employee@company.com'] * 5,
        'to': ['colleague@company.com',
               'external@partner.com',
               'personal@gmail.com',
               'team@company.com',
               'external@competitor.com'],
        'cc': ['manager@company.com', '', 'team@company.com', '', ''],
        'bcc': ['', '', 'personal@yahoo.com', '', 'external@gmail.com'],
        'size': [1000, 2000, 50000, 1500, 75000],
        'attachments': [1, 2, 4, 1, 5],
        'content': [
            'Regular project update report.',
            'Partnership agreement draft.',
            'Product design specifications.',
            'Team meeting schedule.',
            'Market research data analysis.'
        ]
    })
}

# Save test files
for filename, df in test_cases.items():
    df.to_csv(f'test_cases/{filename}', index=False)

# Function to evaluate model on test cases
def evaluate_test_cases(rf_model, lstm_model, scaler, tfidf):
    results = {}
    
    for filename, test_df in test_cases.items():
        # Preprocess test data
        test_df['cc'] = test_df['cc'].fillna('')
        test_df['bcc'] = test_df['bcc'].fillna('')
        test_df['date'] = pd.to_datetime(test_df['date'])
        
        # Feature engineering
        test_df['num_recipients'] = (test_df['to'].str.count(';') +
                                   test_df['cc'].str.count(';') +
                                   test_df['bcc'].str.count(';') + 1)
        test_df['hour'] = test_df['date'].dt.hour
        test_df['day_of_week'] = test_df['date'].dt.dayofweek
        test_df['is_weekend'] = test_df['day_of_week'].isin([5, 6]).astype(int)
        test_df['is_night'] = ((test_df['hour'] < 6) | (test_df['hour'] > 22)).astype(int)
        
        # TF-IDF transformation
        content_tfidf = tfidf.transform(test_df['content']).toarray()
        content_tfidf_df = pd.DataFrame(content_tfidf, columns=tfidf.get_feature_names_out())
        
        # Combine features
        features = ['size', 'attachments', 'num_recipients', 'hour', 'day_of_week',
                   'is_weekend', 'is_night']
        X_test = pd.concat([test_df[features].reset_index(drop=True),
                          content_tfidf_df.reset_index(drop=True)], axis=1)
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Get predictions
        rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        lstm_pred = lstm_model.predict(X_test_lstm)[:, 1]
        
        results[filename] = {
            'Random Forest': rf_pred,
            'LSTM': lstm_pred,
            'content': test_df['content'].values,
            'date': test_df['date'].values,
            'size': test_df['size'].values
        }
    
    return results

# Print and visualize results
def display_results(results):
    for filename, predictions in results.items():
        print(f"\nResults for {filename}:")
        print("-" * 50)
        
        for i in range(len(predictions['content'])):
            print(f"\nEmail {i+1}:")
            print(f"Content: {predictions['content'][i]}")
            print(f"Date: {predictions['date'][i]}")
            print(f"Size: {predictions['size'][i]} bytes")
            print(f"Threat Probability:")
            print(f"  Random Forest: {predictions['Random Forest'][i]:.3f}")
            print(f"  LSTM: {predictions['LSTM'][i]:.3f}")
            
        # Visualize predictions
        plt.figure(figsize=(10, 5))
        x = range(len(predictions['content']))
        plt.plot(x, predictions['Random Forest'], 'bo-', label='Random Forest')
        plt.plot(x, predictions['LSTM'], 'ro-', label='LSTM')
        plt.title(f'Threat Predictions - {filename}')
        plt.xlabel('Email Index')
        plt.ylabel('Threat Probability')
        plt.legend()
        plt.grid(True)
        plt.xticks(x)
        plt.show()

# Usage example:
"""
# After training your models:
results = evaluate_test_cases(rf, lstm_model, scaler, tfidf)
display_results(results)
"""
