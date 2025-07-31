import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import seaborn as sns

class PhishingDetector:

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.context_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        self.text_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.feature_names = []

    def create_sample_dataset(self, n_samples=500): #aim to implement the kaggle dataset soon
        np.random.seed(42)
        
        legit_contexts = [
            "Weekly Newsletter - Company Updates",
            "Your Order Confirmation #12345",
            "Meeting Reminder: Project Review Tomorrow",
            "Monthly Bank Statement Available",
            "Password Reset Confirmation",
            "Welcome to Our Service!",
            "Team Lunch Invitation",
            "System Maintenance Notice",
            "Your Receipt from Apple Store",
            "Flight Confirmation - United Airlines"
        ]
        
        legit_texts = [
            "Hi there! Here's your weekly update with company news and announcements.",
            "Thank you for your purchase. Your order has been confirmed and will ship soon.",
            "This is a reminder about tomorrow's project review meeting at 2 PM.",
            "Your monthly statement is now available in your online banking account.",
            "Your password has been successfully reset as requested.",
            "Welcome! We're excited to have you as part of our community.",
            "You're invited to join us for lunch this Friday at 12:30 PM.",
            "Scheduled maintenance will occur this weekend from 2-4 AM.",
            "Thank you for your purchase. Your receipt is attached for your records.",
            "Your flight is confirmed. Check-in opens 24 hours before departure."
        ]
        
        legit_urls = [
            "https://company.com/newsletter",
            "https://store.amazon.com/orders",
            "https://teams.microsoft.com/meeting",
            "https://bankofamerica.com/statements",
            "https://accounts.google.com/reset",
            "https://service.com/welcome",
            "https://calendar.office.com/lunch",
            "https://status.company.com/maintenance",
            "https://apple.com/receipt",
            "https://united.com/confirmation"
        ]
        
        phish_contexts = [
            "URGENT: Account Suspended - Action Required",
            "You've Won $1,000,000 in Our Lottery!",
            "Security Alert: Suspicious Login Detected",
            "PayPal Account Limited - Verify Now",
            "Final Notice: Payment Failed",
            "Congratulations! You're Our Winner!",
            "Bank Alert: Account Will Be Closed",
            "Microsoft Security Warning",
            "Free iPhone - Claim Your Prize",
            "Tax Refund Waiting - Claim Now"
        ]
        
        phish_texts = [
            "Your account will be suspended in 24 hours! Click here immediately to verify your information or lose access forever!",
            "CONGRATULATIONS! You have won our international lottery! Click here to claim your $1,000,000 prize before it expires!",
            "We detected suspicious activity on your account from Russia. Click here immediately to secure your account!",
            "Your PayPal account has been limited due to security concerns. Update your information now to restore access!",
            "FINAL NOTICE: Your payment has failed multiple times. Update your billing information immediately or face penalties!",
            "You are today's lucky winner! Click here now to claim your amazing prize before someone else takes it!",
            "Your bank account will be permanently closed tomorrow! Confirm your identity immediately to prevent closure!",
            "Microsoft Security Team: Your computer is infected with viruses! Download our tool immediately to remove threats!",
            "FREE iPhone 14 Pro! You're winner number 3 today! Click here to claim your free phone before it's gone!",
            "IRS Notice: You have a tax refund of $2,847 waiting. Click here to claim it before the deadline expires!"
        ]
        
        phish_urls = [
            "http://account-verify.suspicious-site.tk",
            "https://lottery-winner.fake-site.ml",
            "http://192.168.1.100/security-alert",
            "https://paypal-verify.scam-site.ga",
            "http://billing-update.phishing.cf",
            "https://claim-prize.fake-lottery.tk",
            "http://bank-verify.suspicious.ml",
            "https://microsoft-security.fake.ga",
            "http://free-iphone.scam.tk",
            "https://tax-refund.fake-irs.ml"
        ]
        
        emails = []
        
        for i in range(n_samples // 2):
            context = np.random.choice(legit_contexts)
            text = np.random.choice(legit_texts)
            url = np.random.choice(legit_urls)
            emails.append([context, text, url, 0])  # 0 = legitimate
        
        for i in range(n_samples // 2):
            context = np.random.choice(phish_contexts)
            text = np.random.choice(phish_texts)
            url = np.random.choice(phish_urls)
            emails.append([context, text, url, 1])  # 1 = phishing
        
        df = pd.DataFrame(emails, columns=['context', 'text', 'url', 'is_phishing'])
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle rows
                
        return df

    def extract_url_features(self, url):
        url = str(url).lower()
        
        features = []
        
        features.append(len(url))
        
        features.append(1 if url.startswith('https://') else 0)
        
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        features.append(1 if re.search(ip_pattern, url) else 0)
        
        features.append(url.count('.'))
        
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf']
        features.append(1 if any(tld in url for tld in suspicious_tlds) else 0)
        
        #Placeholder for Google safe browsing API 
        #features.append(self.check_safe_browsing(url))
        
        return features

    #TODO: implement google safe browsing API

    def prepare_features(self, df):        
        context_tfidf = self.context_vectorizer.fit_transform(df['context']).toarray()
        
        text_tfidf = self.text_vectorizer.fit_transform(df['text']).toarray()
        
        url_features = []
        for url in df['url']:
            url_features.append(self.extract_url_features(url))
        url_features = np.array(url_features)
        
        all_features = np.hstack([context_tfidf, text_tfidf, url_features])
        
        context_names = [f'context_tfidf_{i}' for i in range(context_tfidf.shape[1])]
        text_names = [f'text_tfidf_{i}' for i in range(text_tfidf.shape[1])]
        url_names = ['url_length', 'has_https', 'has_ip', 'num_dots', 'suspicious_tld', 'is_malicious_api']
        
        self.feature_names = context_names + text_names + url_names
                
        return all_features

    def train_model(self, df):
        
        X = self.prepare_features(df)
        y = df['is_phishing']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"Data:")
        print(f"   • Training samples: {len(X_train)}")
        print(f"   • Testing samples: {len(X_test)}")
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   • Accuracy: {accuracy:.1%}")
        
        results = {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'accuracy': accuracy
        }
        
        return results
    
    def evaluate_model(self, results):
        
        print(f"Overall Accuracy: {results['accuracy']:.1%}")
        
        report = classification_report(results['y_test'], results['y_pred'], 
                               target_names=['Legitimate', 'Phishing'], zero_division=0)

        print(report)
        
        self.plot_confusion_matrix(results['y_test'], results['y_pred'])
        
        self.show_feature_importance()
    
    def plot_confusion_matrix(self, y_test, y_pred):
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legitimate', 'Phishing'],
                    yticklabels=['Legitimate', 'Phishing'])
        
        plt.title('Confusion Matrix - Phishing Detection Model', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        plt.figtext(0.02, 0.02, 
                   'Interpretation: Diagonal values show correct predictions.\n'
                   'Off-diagonal values show classification errors.',
                   fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()
        
        print(f"   • True Negatives (Correct Legitimate): {cm[0,0]}")
        print(f"   • False Positives (Legitimate marked as Phishing): {cm[0,1]}")
        print(f"   • False Negatives (Phishing marked as Legitimate): {cm[1,0]}")
        print(f"   • True Positives (Correct Phishing): {cm[1,1]}")
    
    def show_feature_importance(self):
        importance = self.model.feature_importances_
        
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 Top 10 Most Important Features:")
        for i, row in feature_df.head(10).iterrows():
            print(f"   {row['feature']:<25} {row['importance']:.4f}")
    
    def predict_email(self, context, text, url):
        temp_df = pd.DataFrame({
            'context': [context],
            'text': [text],
            'url': [url]
        })
        
        context_tfidf = self.context_vectorizer.transform(temp_df['context']).toarray()
        text_tfidf = self.text_vectorizer.transform(temp_df['text']).toarray()
        url_features = np.array([self.extract_url_features(url)])
        
        X = np.hstack([context_tfidf, text_tfidf, url_features])
        
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        result = "PHISHING" if prediction == 1 else "LEGITIMATE"
        confidence = f"{max(probability):.1%}"
        
        return f"{result} (Confidence: {confidence})"

def main():
    
    detector = PhishingDetector()
    
    df = detector.create_sample_dataset(500)
    
    print(f"Sample Email Data:")
    sample_data = df.head(5).copy()
    sample_data['type'] = sample_data['is_phishing'].map({0: 'Legitimate', 1: 'Phishing'})
    sample_data['context'] = sample_data['context'].str[:40] + '...'
    sample_data['text'] = sample_data['text'].str[:50] + '...'
    print(sample_data[['context', 'text', 'url', 'type']].to_string(index=False))
    
    results = detector.train_model(df)
    
    detector.evaluate_model(results)
        
    test_cases = [
        {
            'context': 'Team Meeting Tomorrow',
            'text': 'Hi everyone, reminder about our team meeting tomorrow at 2 PM in conference room A.',
            'url': 'https://teams.microsoft.com/meeting'
        },
        {
            'context': 'URGENT: Account Suspended!',
            'text': 'Your account will be suspended in 24 hours! Click here immediately to verify!',
            'url': 'http://account-verify.suspicious-site.tk'
        },
        {
            'context': 'Your Amazon Order Update',
            'text': 'Your recent order has been shipped and will arrive within 2-3 business days.',
            'url': 'https://amazon.com/orders/tracking'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing Email:")
        print(f"   Subject: {case['context']}")
        print(f"   Text: {case['text'][:60]}...")
        print(f"   URL: {case['url']}")
        
        prediction = detector.predict_email(case['context'], case['text'], case['url'])
        print(f"   🔍 Result: {prediction}")
    

#TODO: Google Safe Browsing API Integration Function
if __name__ == "__main__":
    main()

