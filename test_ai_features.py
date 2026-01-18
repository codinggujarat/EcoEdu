import unittest
from ai_service import ai_service
from datetime import datetime
import json
import logging

class MockUser:
    def __init__(self, id, completions):
        self.id = id
        self.challenge_completions = completions

class MockCompletion:
    def __init__(self, completed_at):
        self.completed_at = completed_at

class MockChallenge:
    def __init__(self, id, title, description, category, tags):
        self.id = id
        self.title = title
        self.description = description
        self.category = category
        self.tags = json.dumps(tags)
    
    def get_tags(self):
        return json.loads(self.tags)
    
    @property
    def difficulty_score(self):
        return 1.0

class TestAIService(unittest.TestCase):
    def setUp(self):
        # Setup mock challenges
        self.challenges = [
            MockChallenge(1, "Turn off lights", "Save energy by turning off lights", "energy", ["light", "easy"]),
            MockChallenge(2, "Plant a tree", "Plant a tree in your garden", "tree-planting", ["nature", "outdoor"]),
            MockChallenge(3, "Recycle plastic", "Separate plastic waste", "waste", ["recycle", "plastic"]),
            MockChallenge(4, "Compost food", "Compost organic waste", "waste", ["food", "compost"])
        ]
        ai_service['recommendations'].train(self.challenges)
        
    def test_recommendations(self):
        print("\nTesting Recommendations...")
        # User completed ID 3 (Recycle plastic), should recommend ID 4 (Compost food) as both are waste
        recs = ai_service['recommendations'].get_recommendations([3], top_n=1)
        self.assertTrue(len(recs) > 0)
        print(f"User completed 'Recycle plastic'. Recommended: {recs[0].title}")
        # Ideally it recommends "Compost food" or "Plant a tree" depending on TF-IDF, but definitely returns something
        
    def test_fraud_detection(self):
        print("\nTesting Fraud Detection...")
        now = datetime.utcnow()
        # User with 5 completions in last minute
        completions = [MockCompletion(now) for _ in range(5)]
        user = MockUser(1, completions)
        is_suspicious, reason = ai_service['fraud'].check_activity(user, now)
        self.assertTrue(is_suspicious)
        print(f"Fraud Detected: {reason}")
        
    def test_journal_analysis(self):
        print("\nTesting Journal Analysis...")
        text = "I planted a tree and it made me feel very happy and connected to nature. It was a wonderful day."
        result = ai_service['journal'].analyze_entry(text)
        print(f"Journal Analysis: {result}")
        self.assertIn('sentiment_score', result)
        
    def test_tips(self):
        print("\nTesting Eco Tips...")
        tip = ai_service['tips'].generate_tip(None)
        self.assertTrue(isinstance(tip, str))
        print(f"Generated Tip: {tip}")

if __name__ == '__main__':
    unittest.main()
