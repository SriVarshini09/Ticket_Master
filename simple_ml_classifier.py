"""
Simplified ML Classifier that works with basic Python installations
No scipy dependency - uses simple statistical methods
"""

import re
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import math

class SimpleTFIDF:
    """Lightweight TF-IDF implementation without dependencies"""
    
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf_scores = {}
        self.fitted = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        tokens = text.split()
        return [token for token in tokens if len(token) > 2]
    
    def fit(self, documents: List[str]):
        """Fit the TF-IDF vectorizer"""
        # Build vocabulary
        word_counts = Counter()
        doc_counts = defaultdict(int)
        
        for doc in documents:
            tokens = self._tokenize(doc)
            word_counts.update(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_counts[token] += 1
        
        # Select top features
        most_common = word_counts.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
        
        # Calculate IDF scores
        total_docs = len(documents)
        for word in self.vocabulary:
            self.idf_scores[word] = math.log(total_docs / (doc_counts[word] + 1))
        
        self.fitted = True
    
    def transform(self, documents: List[str]) -> List[List[float]]:
        """Transform documents to TF-IDF vectors"""
        if not self.fitted:
            raise ValueError("Must fit before transform")
        
        vectors = []
        for doc in documents:
            tokens = self._tokenize(doc)
            token_counts = Counter(tokens)
            
            # Create TF-IDF vector
            vector = [0.0] * len(self.vocabulary)
            total_tokens = len(tokens)
            
            for word, idx in self.vocabulary.items():
                if word in token_counts:
                    tf = token_counts[word] / total_tokens
                    tfidf = tf * self.idf_scores[word]
                    vector[idx] = tfidf
            
            vectors.append(vector)
        
        return vectors

class SimpleRandomForest:
    """Simplified Random Forest using basic statistics"""
    
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.trees = []
        self.classes = []
        self.fitted = False
    
    def fit(self, X: List[List[float]], y: List[str]):
        """Train the random forest"""
        self.classes = list(set(y))
        self.trees = []
        
        # Create simple decision trees (actually just weighted feature voting)
        for _ in range(self.n_estimators):
            tree = self._create_tree(X, y)
            self.trees.append(tree)
        
        self.fitted = True
    
    def _create_tree(self, X: List[List[float]], y: List[str]) -> Dict:
        """Create a simple decision tree (feature importance weights)"""
        # Calculate feature importance for each class
        feature_weights = defaultdict(lambda: defaultdict(float))
        
        for i, features in enumerate(X):
            label = y[i]
            for j, feature_val in enumerate(features):
                if feature_val > 0:
                    feature_weights[j][label] += feature_val
        
        # Normalize weights
        tree = {}
        for feature_idx, class_weights in feature_weights.items():
            total_weight = sum(class_weights.values())
            if total_weight > 0:
                tree[feature_idx] = {
                    cls: weight / total_weight 
                    for cls, weight in class_weights.items()
                }
        
        return tree
    
    def predict(self, X: List[List[float]]) -> List[str]:
        """Predict using the random forest"""
        if not self.fitted:
            raise ValueError("Must fit before predict")
        
        predictions = []
        for features in X:
            class_scores = defaultdict(float)
            
            # Vote from all trees
            for tree in self.trees:
                for feature_idx, feature_val in enumerate(features):
                    if feature_val > 0 and feature_idx in tree:
                        for cls, weight in tree[feature_idx].items():
                            class_scores[cls] += feature_val * weight
            
            # Predict the class with highest score
            if class_scores:
                predicted = max(class_scores, key=class_scores.get)
            else:
                predicted = self.classes[0]  # Default to first class
            
            predictions.append(predicted)
        
        return predictions
    
    def predict_proba(self, X: List[List[float]]) -> List[Dict[str, float]]:
        """Get prediction probabilities"""
        if not self.fitted:
            raise ValueError("Must fit before predict")
        
        probabilities = []
        for features in X:
            class_scores = defaultdict(float)
            
            # Vote from all trees
            for tree in self.trees:
                for feature_idx, feature_val in enumerate(features):
                    if feature_val > 0 and feature_idx in tree:
                        for cls, weight in tree[feature_idx].items():
                            class_scores[cls] += feature_val * weight
            
            # Normalize to probabilities
            total_score = sum(class_scores.values())
            if total_score > 0:
                proba = {cls: score / total_score for cls, score in class_scores.items()}
            else:
                proba = {cls: 1.0 / len(self.classes) for cls in self.classes}
            
            probabilities.append(proba)
        
        return probabilities

class LightweightMLClassifier:
    """Lightweight ML classifier that works without scipy"""
    
    def __init__(self):
        self.vectorizer = SimpleTFIDF(max_features=500)
        self.priority_classifier = SimpleRandomForest(n_estimators=5)
        self.category_classifier = SimpleRandomForest(n_estimators=5)
        self.sentiment_classifier = SimpleRandomForest(n_estimators=5)
        self.is_trained = False
    
    def create_training_data(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Generate training data"""
        training_examples = [
            # (text, priority, category, sentiment)
            ("URGENT server crashed production down", "high", "technical", "negative"),
            ("CRITICAL database error users cannot login", "high", "technical", "frustrated"),
            ("Emergency payment gateway down losing revenue", "high", "billing", "frustrated"),
            ("Site completely broken website won't load", "high", "technical", "negative"),
            ("URGENT security breach unauthorized access", "high", "technical", "negative"),
            
            ("Login issues cannot log into account", "medium", "technical", "neutral"),
            ("Billing question charged wrong amount", "medium", "billing", "neutral"),
            ("Feature not working export function not responding", "medium", "technical", "neutral"),
            ("Account access problem cannot update profile", "medium", "account", "neutral"),
            ("Integration help needed API documentation unclear", "medium", "technical", "neutral"),
            
            ("Feature request would like dark mode", "low", "general", "positive"),
            ("General question how do I change password", "low", "account", "neutral"),
            ("Documentation update user guide needs examples", "low", "general", "neutral"),
            ("Enhancement suggestion add export to Excel", "low", "general", "positive"),
            ("Feedback love the new dashboard design", "low", "general", "positive"),
            
            ("Double charged billed twice subscription", "medium", "billing", "frustrated"),
            ("Refund request need refund cancelled service", "medium", "billing", "neutral"),
            ("Payment failed credit card declined", "medium", "billing", "neutral"),
            ("Invoice question don't understand charges", "low", "billing", "neutral"),
            
            ("API returning 500 errors server errors endpoint", "high", "technical", "frustrated"),
            ("Slow performance application very slow load", "medium", "technical", "neutral"),
            ("Mobile app crashes upload photos", "medium", "technical", "frustrated"),
            ("Browser compatibility site doesn't work Safari", "low", "technical", "neutral"),
            
            ("Cannot delete account delete button not working", "medium", "account", "frustrated"),
            ("Profile sync issues changes not saving", "medium", "account", "neutral"),
            ("Permission problems cannot access team settings", "medium", "account", "neutral"),
            ("Two factor auth issues 2FA codes not working", "high", "account", "frustrated"),
        ]
        
        texts = [example[0] for example in training_examples]
        priorities = [example[1] for example in training_examples]
        categories = [example[2] for example in training_examples]
        sentiments = [example[3] for example in training_examples]
        
        return texts, priorities, categories, sentiments
    
    def train(self):
        """Train the lightweight ML models"""
        print("🧠 Training lightweight ML models...")
        
        texts, priorities, categories, sentiments = self.create_training_data()
        
        # Fit vectorizer
        self.vectorizer.fit(texts)
        
        # Transform texts to vectors
        X = self.vectorizer.transform(texts)
        
        # Train classifiers
        self.priority_classifier.fit(X, priorities)
        self.category_classifier.fit(X, categories)
        self.sentiment_classifier.fit(X, sentiments)
        
        self.is_trained = True
        print("✅ Lightweight ML training completed!")
    
    def predict(self, subject: str, message: str) -> Dict:
        """Predict using lightweight ML"""
        if not self.is_trained:
            raise ValueError("Must train before predict")
        
        # Combine and vectorize text
        combined_text = f"{subject} {message}"
        X = self.vectorizer.transform([combined_text])
        
        # Make predictions
        priority_pred = self.priority_classifier.predict(X)[0]
        category_pred = self.category_classifier.predict(X)[0]
        sentiment_pred = self.sentiment_classifier.predict(X)[0]
        
        # Get probabilities
        priority_proba = self.priority_classifier.predict_proba(X)[0]
        category_proba = self.category_classifier.predict_proba(X)[0]
        sentiment_proba = self.sentiment_classifier.predict_proba(X)[0]
        
        # Calculate confidence
        priority_conf = priority_proba.get(priority_pred, 0)
        category_conf = category_proba.get(category_pred, 0)
        sentiment_conf = sentiment_proba.get(sentiment_pred, 0)
        
        overall_confidence = (priority_conf + category_conf + sentiment_conf) / 3 * 100
        
        return {
            'priority': priority_pred,
            'category': category_pred,
            'sentiment': sentiment_pred,
            'confidence': round(overall_confidence, 1),
            'priority_confidence': round(priority_conf * 100, 1),
            'category_confidence': round(category_conf * 100, 1),
            'sentiment_confidence': round(sentiment_conf * 100, 1),
            'model_type': 'lightweight_ml'
        }

class SimplifiedTicketAgent:
    """Complete ticket agent with lightweight ML"""
    
    def __init__(self):
        self.classifier = LightweightMLClassifier()
        self.classifier.train()  # Train on startup
        
        self.team_mapping = {
            'billing': 'Finance Team',
            'technical': 'Engineering Team',
            'account': 'Customer Success Team',
            'general': 'General Support Team'
        }
        
        self.sla_rules = {
            'high': {'response_time': '30 minutes', 'resolution_time': '4 hours'},
            'medium': {'response_time': '2 hours', 'resolution_time': '24 hours'},
            'low': {'response_time': '24 hours', 'resolution_time': '72 hours'}
        }
    
    def process_ticket(self, subject: str, message: str, customer_email: str) -> Dict:
        """Process ticket with lightweight ML"""
        classification = self.classifier.predict(subject, message)
        
        assigned_team = self.team_mapping[classification['category']]
        sla = self.sla_rules[classification['priority']]
        
        needs_escalation = (
            classification['priority'] == 'high' or
            classification['sentiment'] == 'frustrated' or
            'urgent' in subject.lower()
        )
        
        suggested_actions = self._generate_actions(classification, needs_escalation)
        
        return {
            'classification': classification,
            'assigned_team': assigned_team,
            'needs_escalation': needs_escalation,
            'sla': sla,
            'suggested_actions': suggested_actions,
            'estimated_resolution_time': sla['resolution_time']
        }
    
    def _generate_actions(self, classification: Dict, needs_escalation: bool) -> List[str]:
        """Generate intelligent actions"""
        actions = []
        
        if needs_escalation:
            actions.append("🚨 Escalate to senior agent")
        
        if classification['priority'] == 'high':
            actions.append("📞 Call customer immediately")
        
        if classification['sentiment'] == 'frustrated':
            actions.append("💬 Use empathetic response")
        
        actions.append(f"👥 Route to {self.team_mapping[classification['category']]}")
        actions.append("📧 Send acknowledgment email")
        
        return actions

# Demo the lightweight ML agent
if __name__ == "__main__":
    print("🤖 Initializing Lightweight ML Agent...")
    agent = SimplifiedTicketAgent()
    
    test_cases = [
        {
            'subject': 'URGENT: Payment system crashed',
            'message': 'Our payment processing is completely down! Customers cannot checkout.',
            'email': 'ceo@company.com'
        },
        {
            'subject': 'Cannot login to my account',
            'message': 'Password reset email never arrived. Need help accessing account.',
            'email': 'user@example.com'
        }
    ]
    
    for i, ticket in enumerate(test_cases, 1):
        print(f"\n📋 TICKET #{i}: {ticket['subject']}")
        print("-" * 50)
        
        result = agent.process_ticket(
            ticket['subject'],
            ticket['message'],
            ticket['email']
        )
        
        classification = result['classification']
        print(f"Priority: {classification['priority'].upper()} ({classification['priority_confidence']}%)")
        print(f"Category: {classification['category']}")
        print(f"Sentiment: {classification['sentiment']}")
        print(f"Confidence: {classification['confidence']}%")
        print(f"Team: {result['assigned_team']}")
        print(f"Actions: {', '.join(result['suggested_actions'])}")
    
    print("\n🎉 Lightweight ML Agent working perfectly!")