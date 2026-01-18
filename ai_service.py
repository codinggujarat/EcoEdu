import json
import random
import numpy as np
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import hashlib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.challenges = []
        self.challenge_indices = {}

    def train(self, challenges):
        """
        Train the recommendation model on the list of challenges.
        challenges: List of Challenge objects (or dicts)
        """
        self.challenges = challenges
        if not challenges:
            return

        # Create content string for each challenge: title + description + category + tags
        content_corpus = []
        for c in challenges:
            tags = " ".join(c.get_tags()) if hasattr(c, 'get_tags') else ""
            content = f"{c.title} {c.description} {c.category} {tags}"
            content_corpus.append(content)
            self.challenge_indices[c.id] = len(content_corpus) - 1

        self.tfidf_matrix = self.vectorizer.fit_transform(content_corpus)
        logger.info(f"Recommendation engine trained on {len(challenges)} challenges.")

    def get_recommendations(self, user_history, top_n=3):
        """
        Get recommendations based on user's completed challenges.
        user_history: List of challenge_ids that the user has completed.
        """
        if self.tfidf_matrix is None:
            return []

        # If user has no history, return simple difficulty-based or random recommendations
        # For now, just return random easy ones
        if not user_history:
            simple_recs = [c for c in self.challenges if c.difficulty_score <= 2.0]
            if not simple_recs:
                simple_recs = self.challenges
            return random.sample(simple_recs, min(len(simple_recs), top_n))

        # Content-Based Filtering
        # 1. Aggregate user profile vector (mean of completed challenge vectors)
        user_indices = [self.challenge_indices[cid] for cid in user_history if cid in self.challenge_indices]
        
        if not user_indices:
             return random.sample(self.challenges, min(len(self.challenges), top_n))

        user_profile = np.asarray(self.tfidf_matrix[user_indices].mean(axis=0))
        
        # Reshape for sklearn
        user_profile = user_profile.reshape(1, -1)

        # 2. Compute cosine similarity between user profile and all challenges
        cosine_sim = cosine_similarity(user_profile, self.tfidf_matrix)

        # 3. Get top N similar challenges
        # Flatten and sort
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Filter out already completed challenges
        rec_challenges = []
        for idx, score in sim_scores:
            challenge = self.challenges[idx]
            if challenge.id not in user_history:
                rec_challenges.append(challenge)
                if len(rec_challenges) >= top_n:
                    break
        
        return rec_challenges

class VerificationSystem:
    def __init__(self):
        self.model = None
        try:
            import tensorflow as tf
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
            from tensorflow.keras.preprocessing import image
            self.tf_image = image
            self.preprocess_input = preprocess_input
            self.decode_predictions = decode_predictions
            self.model = MobileNetV2(weights='imagenet', include_top=True)
            logger.info("MobileNetV2 loaded successfully.")
        except ImportError:
            logger.warning("TensorFlow not found. AI Verification disabled.")
        except Exception as e:
            logger.warning(f"Failed to load MobileNetV2: {e}")

    def verify_submission(self, image_path, challenge_type):
        if not self.model:
             # Fallback mock for testing if model fails to load
             return False, 0.0, "Model not generated"
        
        try:
            img = self.tf_image.load_img(image_path, target_size=(224, 224))
            x = self.tf_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = self.preprocess_input(x)
            
            preds = self.model.predict(x)
            decoded = self.decode_predictions(preds, top=3)[0]
            
            # Simple keyword matching for MVP
            # challenge_type: 'tree-planting', 'waste', 'energy'
            keywords = {
                'tree-planting': ['plant', 'pot', 'flower', 'tree', 'garden', 'leaf', 'grass'],
                'waste': ['trash', 'bin', 'plastic', 'paper', 'container', 'bottle', 'box'],
                'energy': ['light', 'lamp', 'switch', 'dark'] # Hard to verify energy saving by photo
            }
            
            target_keywords = keywords.get(challenge_type, [])
            
            matched = False
            confidence = 0.0
            
            logger.info(f"Predictions for {challenge_type}: {decoded}")
            
            for (id, label, prob) in decoded:
                for k in target_keywords:
                    if k in label.lower():
                        matched = True
                        confidence = float(prob)
                        break
                if matched: 
                    break
            
            return matched, confidence, "Verified" if matched else "Subject mismatch"
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return False, 0.0, str(e)

class EcoTipGenerator:
    def generate_tip(self, user):
        """
        Generate a context-aware eco-tip.
        """
        tips = [
            "Turn off lights when leaving a room to save energy.",
            "Use a reusable water bottle to reduce plastic waste.",
            "Plant a tree in your local community.",
            "Compost your food scraps.",
            "Walk or bike instead of driving properly.",
            "Unplug electronics when not in use.",
            "Use cold water for laundry to save energy.",
            "Buy local produce to reduce carbon footprint."
        ]
        # TODO: Use user location/weather if available
        return random.choice(tips)

class FraudDetector:
    def check_activity(self, user, submission_time):
        """
        Check for suspicious activity (velocity check).
        user: User object
        submission_time: datetime
        Returns: (is_suspicious: bool, reason: str)
        """
        # 1. High Velocity Check: > 3 submissions in 10 minutes
        recent_submissions = [c for c in user.challenge_completions 
                              if c.completed_at > submission_time - timedelta(minutes=10)]
        
        if len(recent_submissions) >= 3:
            return True, "High velocity: Too many submissions in short time."
            
        return False, None

class JournalAnalyzer:
    def __init__(self):
        try:
            from textblob import TextBlob
            self.TextBlob = TextBlob
        except:
            self.TextBlob = None

    def analyze_entry(self, text):
        """
        Analyze journal text.
        """
        if not self.TextBlob or not text:
             return {"sentiment": "neutral", "quality": "unknown", "feedback": "Keep writing!"}
             
        blob = self.TextBlob(text)
        sentiment = blob.sentiment.polarity # -1 to 1
        
        quality = "good"
        feedback = "Great reflection!"
        
        if len(text.split()) < 10:
            quality = "poor"
            feedback = "Try to elaborate more on your experience."
        elif sentiment < -0.5:
            quality = "concerned"
            feedback = "It seems you had a tough time. Don't give up!"
        elif sentiment > 0.5:
            quality = "excellent"
            feedback = "Wonderful positive reflection!"
            
        return {
            "sentiment_score": sentiment,
            "quality": quality,
            "feedback": feedback
        }


class AutoActionValidator:
    def __init__(self):
        self.duplicate_detector = DuplicateImageDetector()
        
    def calculate_user_trust_score(self, user):
        """
        Calculate user trust score based on their history.
        Returns a float between 0 and 1.
        """
        if not hasattr(user, 'challenge_completions') or not user.challenge_completions:
            return 0.5  # Default medium trust for new users
            
        # Calculate trust based on successful completions vs flagged/rejected submissions
        total_submissions = len(user.challenge_completions)
        successful_completions = sum(1 for c in user.challenge_completions 
                                   if getattr(c, 'status', 'approved') == 'approved')
        
        trust_score = successful_completions / total_submissions
        
        # Cap the score between 0 and 1
        return max(0.0, min(1.0, trust_score))
        
    def validate_submission(self, image_path, challenge_type, user):
        """
        Main validation function that makes auto-decision based on multiple factors.
        
        Args:
            image_path: Path to the submitted image
            challenge_type: Type of challenge being submitted
            user: User object containing history
        
        Returns:
            dict: {"decision": "APPROVED | REVIEW | REJECTED", "confidence": <float>, "reason": "<explanation>"}
        """
        current_time = datetime.now()
        
        # Step 1: Get AI verification results
        verification_result = self.verify_image(image_path, challenge_type)
        ai_confidence = verification_result['confidence']
        verification_success = verification_result['success']
        verification_message = verification_result['message']
        
        # Step 2: Check for exact duplicate images
        is_duplicate = self.duplicate_detector.is_duplicate(image_path)
        
        # Step 3: Check for similar images (potential cheating)
        is_similar, similarity_score, similar_image_path = self.duplicate_detector.is_similar(image_path, threshold=0.85)
        
        # Step 4: Check for frequency anomalies (cheating patterns)
        freq_anomaly, submission_count, freq_msg = self.duplicate_detector.detect_frequency_anomaly(user, current_time, time_window_minutes=10)
        
        # Step 5: Get fraud detection result
        fraud_result = self.check_fraud(user, current_time)
        is_suspicious = fraud_result[0]
        fraud_reason = fraud_result[1]
        
        # Step 6: Calculate user trust score
        user_trust_score = self.calculate_user_trust_score(user)
        
        # Apply decision logic according to requirements
        # Priority order matters here
        
        # First check: if it's an exact duplicate image, reject immediately
        if is_duplicate:
            # Auto-deduct points (soft penalty)
            self._apply_soft_penalty(user)
            # Decision affects civic score, rewards, and certificates
            self._update_user_metrics(user, "rejected")
            return {
                "decision": "REJECTED",
                "confidence": 0.0,
                "reason": "Exact duplicate image detected. Submission rejected automatically. Points deducted."
            }
        
        # Second check: if similar image detected (potential cheating)
        if is_similar:
            # Auto-warn user and deduct points (soft penalty)
            self._apply_soft_penalty(user)
            # Decision affects civic score, rewards, and certificates
            self._update_user_metrics(user, "rejected")
            return {
                "decision": "REJECTED",
                "confidence": 0.0,
                "reason": f"Similar image detected (match: {similarity_score:.2f}). Potential cheating. Submission rejected and points deducted."
            }
        
        # Third check: if frequency anomaly detected
        if freq_anomaly:
            # Flag for review due to suspicious activity pattern
            # Decision affects civic score, rewards, and certificates
            self._update_user_metrics(user, "review")
            return {
                "decision": "REVIEW",
                "confidence": ai_confidence,
                "reason": f"Frequency anomaly detected: {freq_msg}. Requires admin review."
            }
        
        # Fourth check: if suspicious activity, flag for review
        if is_suspicious:
            # Decision affects civic score, rewards, and certificates
            self._update_user_metrics(user, "review")
            return {
                "decision": "REVIEW",
                "confidence": ai_confidence,
                "reason": f"Suspicious activity detected: {fraud_reason}. Requires admin review."
            }
        
        # Main decision logic based on confidence and trust score
        if ai_confidence >= 0.85 and user_trust_score >= 0.6:
            # Auto-approve and award points immediately
            self._award_points(user)
            self._update_user_metrics(user, "approved")
            return {
                "decision": "APPROVED",
                "confidence": ai_confidence,
                "reason": f"High confidence ({ai_confidence:.2f}) and trusted user ({user_trust_score:.2f}). Verification: {verification_message}."
            }
        elif 0.60 <= ai_confidence < 0.85:
            # Decision affects civic score, rewards, and certificates
            self._update_user_metrics(user, "review")
            return {
                "decision": "REVIEW",
                "confidence": ai_confidence,
                "reason": f"Medium confidence ({ai_confidence:.2f}). Requires admin review."
            }
        elif ai_confidence < 0.60:
            # Auto-reject and provide learning feedback
            self._update_user_metrics(user, "rejected")
            return {
                "decision": "REJECTED",
                "confidence": ai_confidence,
                "reason": f"Low confidence ({ai_confidence:.2f}). Rejected automatically. {verification_message}."
            }
        else:
            # Fallback case
            self._update_user_metrics(user, "review")
            return {
                "decision": "REVIEW",
                "confidence": ai_confidence,
                "reason": "Unable to make automatic decision. Requires admin review."
            }
            
    def verify_image(self, image_path, challenge_type):
        """
        Verify the image using the existing verification system.
        """
        # This uses the existing VerificationSystem
        verification_system = VerificationSystem()
        matched, confidence, message = verification_system.verify_submission(image_path, challenge_type)
        
        return {
            "success": matched,
            "confidence": confidence,
            "message": message
        }
        
    def check_fraud(self, user, submission_time):
        """
        Check for fraudulent activity using the existing fraud detector.
        """
        # This uses the existing FraudDetector
        fraud_detector = FraudDetector()
        return fraud_detector.check_activity(user, submission_time)
    
    def _update_user_metrics(self, user, status):
        """
        Update user metrics based on submission status.
        Affects civic score, rewards, and certificates.
        """
        # In a real implementation, this would connect to the database
        # and update the user's civic score, rewards, and certificates
        if hasattr(user, 'civic_score'):
            if status == "approved":
                user.civic_score = getattr(user, 'civic_score', 0) + 10
            elif status == "rejected":
                user.civic_score = max(0, getattr(user, 'civic_score', 0) - 5)
            # For review, we don't change the score until a final decision
        
        # Update submission status tracking
        if hasattr(user, 'submission_status_count'):
            if status not in user.submission_status_count:
                user.submission_status_count[status] = 0
            user.submission_status_count[status] += 1
        
        logger.info(f"Updated user metrics for status: {status}")
    
    def _award_points(self, user):
        """
        Award points to user for approved submission.
        """
        if hasattr(user, 'points'):
            user.points = getattr(user, 'points', 0) + 10  # Standard points for approval
        
        # In a real implementation, we would also generate/update certificates
        if hasattr(user, 'certificates'):
            # Add or update certificates based on completed challenges
            pass
        
        logger.info("Points awarded for approved submission")
    
    def _apply_soft_penalty(self, user):
        """
        Apply a soft penalty to user for detected cheating attempts.
        """
        # Deduct points as soft penalty
        if hasattr(user, 'points'):
            current_points = getattr(user, 'points', 0)
            penalty_amount = min(5, current_points)  # Max 5 points deduction, or current points if less
            user.points = current_points - penalty_amount
            
        # Decrease trust score slightly
        if hasattr(user, 'trust_score'):
            user.trust_score = max(0.1, user.trust_score - 0.1)  # Minimum 0.1 trust score
        
        logger.info(f"Soft penalty applied to user: {penalty_amount} points deducted")




class PersonalizedChallengeGenerator:
    def __init__(self):
        self.challenge_templates = {
            "recycling": [
                "You often recycle {item}. Try reducing {item} consumption for 3 days.",
                "Since you're good at {action}, challenge yourself with {related_action}.",
                "Your {activity} habit is great! Now try {new_activity}.",
                "Continue your {habit} streak for {duration} more days.",
                "Take your {behavior} to the next level with {advanced_behavior}."
            ],
            "energy": [
                "Try reducing energy usage during peak hours for {duration} days.",
                "Challenge yourself to {action} for {duration} days.",
                "Since you care about {issue}, try {solution}.",
                "Build on your {activity} success with {extension}.",
                "Level up your {behavior} routine with {enhancement}."
            ],
            "transportation": [
                "Try {alternative_transport} instead of driving for {duration} days.",
                "Challenge yourself to {eco_action} for {duration} days.",
                "Since you're committed to {behavior}, try {next_step}.",
                "Continue your {pattern} with {enhancement}.",
                "Expand your {activity} with {additional_activity}."
            ],
            "general": [
                "Try {action} for {duration} days to build on your progress.",
                "Since you're good at {activity}, try {advanced_activity}.",
                "Challenge yourself with {behavior} for {duration} days.",
                "Continue your {habit} streak with {enhancement}.",
                "Build on your {progress} with {next_step}."
            ]
        }
        
        self.activity_patterns = {
            "bottles": ["bottle", "plastic", "container"],
            "bags": ["bag", "plastic bag", "shopping bag"],
            "energy": ["light", "electricity", "power", "energy"],
            "transport": ["car", "driving", "bus", "bike", "walking", "public transport"],
            "waste": ["waste", "garbage", "trash", "compost", "recycle"]
        }
    
    def generate_personalized_challenge(self, user):
        """
        Generate a personalized challenge based on user's past actions, location, and civic score.
        
        Args:
            user: User object containing history and attributes
        
        Returns:
            dict: Personalized challenge with title, description, and difficulty
        """
        # Analyze user's past actions
        past_actions = self._analyze_past_actions(user)
        
        # Determine user's civic score level
        civic_score = getattr(user, 'civic_score', 0)
        
        # Get user's location if available
        user_location = getattr(user, 'location', None)
        
        # Generate challenge based on analysis
        challenge = self._create_challenge(past_actions, civic_score, user_location, user)
        
        return challenge
    
    def _analyze_past_actions(self, user):
        """
        Analyze user's past actions to identify patterns and preferences.
        """
        if not hasattr(user, 'challenge_completions') or not user.challenge_completions:
            return {
                "most_common_category": "general",
                "frequent_activities": [],
                "preferred_difficulty": "medium",
                "patterns": []
            }
        
        # Count categories and activities
        category_counts = {}
        activity_counts = {}
        total_points = 0
        
        for completion in user.challenge_completions:
            category = getattr(completion, 'category', 'general').lower()
            title = getattr(completion, 'title', '').lower()
            
            # Count categories
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count specific activities
            activity_counts[title] = activity_counts.get(title, 0) + 1
            
            # Sum points for difficulty assessment
            points = getattr(completion, 'points', 0)
            total_points += points
        
        # Determine most common category
        most_common_category = max(category_counts, key=category_counts.get) if category_counts else "general"
        
        # Determine frequent activities
        frequent_activities = [activity for activity, count in activity_counts.items() if count > 1]
        
        # Determine preferred difficulty based on points
        avg_points = total_points / len(user.challenge_completions) if user.challenge_completions else 0
        if avg_points > 15:
            preferred_difficulty = "hard"
        elif avg_points > 8:
            preferred_difficulty = "medium"
        else:
            preferred_difficulty = "easy"
        
        # Identify patterns
        patterns = []
        for pattern, keywords in self.activity_patterns.items():
            if any(keyword in ' '.join(frequent_activities).lower() for keyword in keywords):
                patterns.append(pattern)
        
        return {
            "most_common_category": most_common_category,
            "frequent_activities": frequent_activities,
            "preferred_difficulty": preferred_difficulty,
            "patterns": patterns,
            "total_completions": len(user.challenge_completions)
        }
    
    def _create_challenge(self, past_actions, civic_score, user_location, user):
        """
        Create a personalized challenge based on user analysis.
        """
        # Determine challenge category
        category = past_actions["most_common_category"]
        if category not in self.challenge_templates:
            category = "general"
        
        # Select template
        templates = self.challenge_templates[category]
        template = templates[0]  # Could implement more sophisticated selection
        
        # Generate dynamic values based on user data
        duration = self._determine_duration(civic_score, past_actions["preferred_difficulty"])
        
        # Get relevant items/actions based on user's patterns
        item = self._get_relevant_item(past_actions)
        action = self._get_relevant_action(past_actions, category)
        
        # Format the challenge text
        challenge_text = template.format(
            item=item,
            action=action,
            duration=duration,
            activity=past_actions.get("frequent_activities", ["eco-friendly activities"])[0] if past_actions.get("frequent_activities") else "eco-friendly activities",
            habit=past_actions.get("frequent_activities", ["sustainable habits"])[0] if past_actions.get("frequent_activities") else "sustainable habits",
            behavior="environmental actions",
            related_action="a related eco-friendly activity",
            new_activity="a new sustainable practice",
            advanced_behavior="an advanced environmental practice",
            issue="environmental conservation",
            solution="a practical solution",
            enhancement="an enhancement to your current routine",
            alternative_transport="public transport or biking",
            eco_action="an eco-friendly action",
            next_step="the next step in your environmental journey",
            pattern="your established pattern",
            progress="your environmental progress"
        )
        
        # Determine difficulty based on user's level
        difficulty = self._determine_difficulty(civic_score, past_actions["preferred_difficulty"])
        
        # Calculate estimated points
        points = self._estimate_points(difficulty)
        
        return {
            "title": f"Personal Challenge: {action.title()} for {duration} Days",
            "description": challenge_text,
            "category": category,
            "difficulty": difficulty,
            "estimated_duration_days": duration,
            "estimated_points": points,
            "personalized_for_user": True,
            "generated_from_analysis": past_actions,
            "location_based": user_location is not None
        }
    
    def _determine_duration(self, civic_score, preferred_difficulty):
        """
        Determine challenge duration based on user's civic score and preferred difficulty.
        """
        base_duration = 3  # Default to 3 days
        
        # Adjust based on civic score
        if civic_score > 50:
            base_duration = 7
        elif civic_score > 20:
            base_duration = 5
        
        # Adjust based on difficulty preference
        if preferred_difficulty == "easy":
            base_duration = max(2, base_duration - 1)
        elif preferred_difficulty == "hard":
            base_duration = min(14, base_duration + 2)
        
        return base_duration
    
    def _get_relevant_item(self, past_actions):
        """
        Get a relevant item based on user's past actions.
        """
        if past_actions["frequent_activities"]:
            # Extract common items from past activities
            activities_str = ' '.join(past_actions["frequent_activities"]).lower()
            for pattern, keywords in self.activity_patterns.items():
                for keyword in keywords:
                    if keyword in activities_str:
                        return keyword
        
        # Default items based on patterns
        if past_actions["patterns"]:
            pattern = past_actions["patterns"][0]
            return self.activity_patterns.get(pattern, ["eco-friendly item"])[0]
        
        return "eco-friendly items"
    
    def _get_relevant_action(self, past_actions, category):
        """
        Get a relevant action based on user's past actions and category.
        """
        if past_actions["frequent_activities"]:
            # Use the most recent activity as inspiration
            recent_activity = past_actions["frequent_activities"][-1]
            if recent_activity:
                return recent_activity
        
        # Default actions based on category
        default_actions = {
            "recycling": "reduce waste",
            "energy": "save energy",
            "transportation": "use eco-friendly transport",
            "general": "practice sustainability"
        }
        
        return default_actions.get(category, "practice sustainability")
    
    def _determine_difficulty(self, civic_score, preferred_difficulty):
        """
        Determine challenge difficulty based on user's civic score and preferences.
        """
        if civic_score > 50:
            # High score users can handle harder challenges
            if preferred_difficulty == "hard":
                return "hard"
            else:
                return "medium"
        elif civic_score > 20:
            # Medium score users
            return preferred_difficulty
        else:
            # New users get easier challenges
            return "easy"
    
    def _estimate_points(self, difficulty):
        """
        Estimate points based on challenge difficulty.
        """
        point_mapping = {
            "easy": 5,
            "medium": 10,
            "hard": 15
        }
        return point_mapping.get(difficulty, 10)


class SmartNudgeEngine:
    def __init__(self):
        self.nudge_templates = [
            "You were active last week 🌱 One small action today keeps your streak alive!",
            "Great job on your recent challenges! Ready for another?",
            "🌱 Small actions, big impact. Try a quick challenge today!",
            "Your eco journey continues! One step at a time.",
            "Missing you! Time to pick up where you left off.",
            "Consistency builds habits. Let's keep that momentum going!",
            "You're making a difference! Don't break your flow now.",
            "🌱 Feeling motivated? Take on a new challenge today!",
            "Remember: every action counts toward a greener future.",
            "Your streak matters! Complete a challenge to keep it going."
        ]
    
    def analyze_user_engagement(self, user):
        """
        Analyze user engagement patterns to identify missed days, streak breaks, and low engagement.
        """
        if not hasattr(user, 'challenge_completions') or not user.challenge_completions:
            return {
                "missed_days": 0,
                "streak_breaks": 0,
                "low_engagement": True,
                "last_active": None
            }
        
        # Sort completions by date
        sorted_completions = sorted(user.challenge_completions, key=lambda x: x.completed_at)
        
        # Calculate days since last activity
        if sorted_completions:
            last_completion = sorted_completions[-1].completed_at
            days_since_last = (datetime.now() - last_completion).days
        else:
            days_since_last = float('inf')
        
        # Calculate streak breaks
        streak_breaks = 0
        missed_days = 0
        
        # Check for patterns in recent activity
        if len(sorted_completions) >= 2:
            for i in range(1, len(sorted_completions)):
                prev_date = sorted_completions[i-1].completed_at.date()
                curr_date = sorted_completions[i].completed_at.date()
                day_diff = (curr_date - prev_date).days
                
                if day_diff > 1:  # More than one day gap indicates missed days
                    missed_days += day_diff - 1
                    streak_breaks += 1
        
        # Determine engagement level
        recent_activity = [c for c in sorted_completions 
                          if c.completed_at >= datetime.now() - timedelta(days=7)]
        low_engagement = len(recent_activity) < 2  # Less than 2 completions in last week
        
        return {
            "missed_days": missed_days,
            "streak_breaks": streak_breaks,
            "low_engagement": low_engagement,
            "days_since_last": days_since_last,
            "last_active": sorted_completions[-1].completed_at if sorted_completions else None
        }
    
    def generate_nudge_message(self, user):
        """
        Generate a personalized nudge message based on user engagement analysis.
        """
        analysis = self.analyze_user_engagement(user)
        
        # Select template based on user behavior
        if analysis["low_engagement"] and analysis["days_since_last"] > 7:
            # User hasn't been active in over a week
            message = "Missing you! Time to pick up where you left off."
        elif analysis["streak_breaks"] > 0 and analysis["days_since_last"] <= 1:
            # Recent streak break
            message = "Consistency builds habits. Let's keep that momentum going!"
        elif analysis["missed_days"] > 2:
            # Multiple missed days
            message = "You're making a difference! Don't break your flow now."
        elif analysis["low_engagement"]:
            # Low engagement pattern
            message = "🌱 Feeling motivated? Take on a new challenge today!"
        else:
            # General encouraging message
            message = self.nudge_templates[0]  # "You were active last week..."
        
        return message
    
    def should_send_nudge(self, user):
        """
        Determine if a nudge should be sent based on user engagement.
        """
        analysis = self.analyze_user_engagement(user)
        
        # Send nudge if:
        # 1. User has low engagement (less than 2 completions in last week)
        # 2. User has been inactive for more than 2 days
        # 3. User has broken a streak recently
        should_nudge = (
            analysis["low_engagement"] or 
            analysis["days_since_last"] > 2 or 
            analysis["streak_breaks"] > 0
        )
        
        # But don't nudge if user is already very active (completed challenge today)
        if analysis["days_since_last"] == 0:
            return False
            
        return should_nudge
    
    def get_nudge_recommendation(self, user):
        """
        Get a complete nudge recommendation including message and timing.
        """
        if not self.should_send_nudge(user):
            return None
        
        message = self.generate_nudge_message(user)
        analysis = self.analyze_user_engagement(user)
        
        return {
            "message": message,
            "should_nudge": True,
            "analysis": analysis,
            "send_time": "immediate"  # In a real system, this could be scheduled
        }



class CivicScoreExplainer:
    def __init__(self):
        self.positive_explanations = [
            "Your score increased because you completed a high-impact {} challenge.",
            "Great work! Your {} submission contributed positively to your civic score.",
            "Your civic score improved thanks to your commitment to {}.",
            "Score increase: You demonstrated environmental responsibility with your {} action.",
            "Well done! Your {} challenge completion boosted your civic score.",
            "Positive impact recorded: Your {} effort raised your civic score.",
            "Civic score increased due to your {} contribution.",
            "Environmental action recognized: {} improved your civic score."
        ]
        
        self.negative_explanations = [
            "Your score decreased because of {}.",
            "Civic score reduced due to {}.",
            "Your civic score dropped because {}.",
            "Negative impact recorded: {} caused a score reduction.",
            "Your score decreased as a result of {}.",
            "Civic score lowered due to {} activity."
        ]
        
        self.neutral_explanations = [
            "Your score remained stable as there were no significant activities.",
            "No major changes to your civic score recently.",
            "Civic score unchanged - keep engaging to improve your score.",
            "Your civic score remains constant. Try completing challenges to increase it."
        ]
    
    def generate_score_explanation(self, user, previous_score=None, current_score=None, activity_log=None):
        """
        Generate an explanation for why the civic score changed.
        
        Args:
            user: User object
            previous_score: Previous civic score (optional)
            current_score: Current civic score (optional)
            activity_log: Recent activity log (optional)
        
        Returns:
            dict: Explanation of score change with transparency
        """
        # Determine scores if not provided
        if previous_score is None:
            previous_score = getattr(user, 'previous_civic_score', getattr(user, 'civic_score', 0))
        if current_score is None:
            current_score = getattr(user, 'civic_score', 0)
        
        # Determine recent activity if not provided
        if activity_log is None:
            activity_log = self._get_recent_activities(user)
        
        # Calculate score difference
        score_diff = current_score - previous_score
        
        # Generate explanation based on score change
        explanation = self._create_explanation(score_diff, activity_log, previous_score, current_score)
        
        return {
            "previous_score": previous_score,
            "current_score": current_score,
            "score_difference": score_diff,
            "explanation": explanation,
            "transparency": True,
            "trust_factor": self._calculate_trust_factor(activity_log)
        }
    
    def _get_recent_activities(self, user):
        """
        Extract recent activities from user's challenge completions.
        """
        if not hasattr(user, 'challenge_completions') or not user.challenge_completions:
            return []
        
        # Get activities from the last 7 days
        recent_activities = []
        for completion in user.challenge_completions:
            if hasattr(completion, 'completed_at') and completion.completed_at >= datetime.now() - timedelta(days=7):
                activity_info = {
                    "challenge_title": getattr(completion, 'title', 'Unknown'),
                    "challenge_category": getattr(completion, 'category', 'General'),
                    "challenge_type": getattr(completion, 'challenge_type', 'general'),
                    "completed_at": completion.completed_at,
                    "points_awarded": getattr(completion, 'points', 0),
                    "status": getattr(completion, 'status', 'completed')
                }
                recent_activities.append(activity_info)
        
        return recent_activities
    
    def _create_explanation(self, score_diff, activity_log, previous_score, current_score):
        """
        Create a human-readable explanation for the score change.
        """
        if score_diff > 0:
            # Positive change
            if activity_log:
                # Find the most impactful recent activity
                impactful_activity = self._find_most_impactful_activity(activity_log)
                if impactful_activity:
                    category = impactful_activity.get('challenge_category', 'environmental').lower()
                    template = self.positive_explanations[0]
                    return template.format(category)
            
            return f"Your score increased from {previous_score} to {current_score} due to your positive environmental actions."
        
        elif score_diff < 0:
            # Negative change
            reason = "inactivity" if not activity_log else "low-quality submission"
            template = self.negative_explanations[0]
            return template.format(reason)
        
        else:
            # No change
            return self.neutral_explanations[0]
    
    def _find_most_impactful_activity(self, activity_log):
        """
        Find the activity that most likely caused the score change.
        """
        if not activity_log:
            return None
        
        # Sort by recency and points
        sorted_activities = sorted(activity_log, 
                                 key=lambda x: (x.get('completed_at', datetime.min), x.get('points_awarded', 0)), 
                                 reverse=True)
        
        return sorted_activities[0] if sorted_activities else None
    
    def _calculate_trust_factor(self, activity_log):
        """
        Calculate a trust factor based on the consistency and quality of activities.
        """
        if not activity_log:
            return 0.5  # Neutral trust
        
        # Calculate based on number of activities and consistency
        num_activities = len(activity_log)
        
        # More activities indicate higher trust, but cap at 1.0
        trust_factor = min(1.0, num_activities * 0.1 + 0.3)
        
        return round(trust_factor, 2)
    
    def get_detailed_breakdown(self, user):
        """
        Provide a detailed breakdown of score changes.
        """
        explanation = self.generate_score_explanation(user)
        
        # Add more details about the breakdown
        details = {
            "summary": explanation,
            "activities": self._get_recent_activities(user),
            "trend": self._determine_trend(user),
            "recommendations": self._generate_recommendations(user)
        }
        
        return details
    
    def _determine_trend(self, user):
        """
        Determine if the user's score trend is positive, negative, or neutral.
        """
        if not hasattr(user, 'challenge_completions') or len(user.challenge_completions) < 2:
            return "insufficient_data"
        
        # Compare recent scores or activities
        recent_completions = [c for c in user.challenge_completions 
                             if hasattr(c, 'completed_at') and c.completed_at >= datetime.now() - timedelta(days=14)]
        
        if len(recent_completions) >= 3:
            return "positive" if len(recent_completions) > 2 else "neutral"
        else:
            return "needs_improvement"
    
    def _generate_recommendations(self, user):
        """
        Generate recommendations to improve the civic score.
        """
        recommendations = []
        
        # Check for missing activities
        recent_activities = self._get_recent_activities(user)
        if not recent_activities:
            recommendations.append("Complete your first challenge to start building your civic score.")
        
        # Suggest consistent activity
        if len(recent_activities) < 3:
            recommendations.append("Try completing challenges consistently to build your score.")
        
        # Suggest variety
        categories = set(a.get('challenge_category', '') for a in recent_activities)
        if len(categories) <= 1 and 'General' not in categories:
            recommendations.append("Try challenges from different categories to diversify your impact.")
        
        return recommendations


class DuplicateImageDetector:
    def __init__(self):
        self.image_hashes = set()  # In production, this should be persistent storage
        self.perceptual_hashes = {}  # Store perceptual hashes for similarity detection
        
    def _calculate_image_hash(self, image_path):
        """
        Calculate a hash for the image to detect exact duplicates.
        """
        try:
            with open(image_path, 'rb') as f:
                img_data = f.read()
                # Create a hash of the image data
                image_hash = hashlib.md5(img_data).hexdigest()
                return image_hash
        except Exception as e:
            logger.error(f"Error calculating image hash: {e}")
            return None
    
    def _calculate_perceptual_hash(self, image_path):
        """
        Calculate a perceptual hash for detecting similar images.
        """
        try:
            # Try to use PIL for image processing if available
            try:
                from PIL import Image
                import imagehash
                
                img = Image.open(image_path)
                # Calculate perceptual hash (pHash)
                phash = imagehash.phash(img)
                return str(phash)
            except ImportError:
                # Fallback to simple hash if PIL/imagehash not available
                logger.warning("PIL or imagehash not available, using basic hash")
                return self._calculate_image_hash(image_path)
        except Exception as e:
            logger.error(f"Error calculating perceptual hash: {e}")
            return None
            
    def is_duplicate(self, image_path):
        """
        Check if the image is an exact duplicate of previously submitted images.
        """
        image_hash = self._calculate_image_hash(image_path)
        if image_hash is None:
            return False  # If we can't calculate hash, assume not duplicate
            
        if image_hash in self.image_hashes:
            return True
        else:
            self.image_hashes.add(image_hash)
            # Also store perceptual hash
            p_hash = self._calculate_perceptual_hash(image_path)
            if p_hash:
                self.perceptual_hashes[image_path] = p_hash
            return False
    
    def is_similar(self, image_path, threshold=0.85):
        """
        Check if the image is similar to previously submitted images using perceptual hashing.
        
        Args:
            image_path: Path to the image to check
            threshold: Similarity threshold (0-1), higher means more similar required
        
        Returns:
            tuple: (is_similar: bool, similarity_score: float, similar_image_path: str)
        """
        try:
            current_phash = self._calculate_perceptual_hash(image_path)
            if not current_phash:
                return False, 0.0, None
            
            # Compare with stored perceptual hashes
            for stored_path, stored_phash in self.perceptual_hashes.items():
                try:
                    # Calculate similarity between hashes
                    similarity = self._calculate_hash_similarity(current_phash, stored_phash)
                    if similarity >= threshold:
                        return True, similarity, stored_path
                except:
                    continue
            
            return False, 0.0, None
        except Exception as e:
            logger.error(f"Error checking image similarity: {e}")
            return False, 0.0, None
    
    def _calculate_hash_similarity(self, hash1, hash2):
        """
        Calculate similarity between two perceptual hashes.
        """
        try:
            # Convert hex strings to integers and XOR them
            h1_int = int(hash1, 16)
            h2_int = int(hash2, 16)
            
            # XOR to find differences
            xor_result = h1_int ^ h2_int
            
            # Count number of different bits
            diff_bits = bin(xor_result).count('1')
            
            # Calculate similarity (0-1, where 1 is identical)
            total_bits = len(hash1) * 4  # Each hex digit represents 4 bits
            similarity = 1.0 - (diff_bits / total_bits)
            
            return similarity
        except:
            # Fallback to simple comparison
            return 1.0 if hash1 == hash2 else 0.0
    
    def detect_frequency_anomaly(self, user, current_time, time_window_minutes=60):
        """
        Detect submission frequency anomalies (too many submissions in short time).
        
        Args:
            user: User object
            current_time: Current timestamp
            time_window_minutes: Time window to check in minutes
        
        Returns:
            tuple: (is_anomaly: bool, submission_count: int, warning_message: str)
        """
        if not hasattr(user, 'submissions') or not user.submissions:
            return False, 0, "No prior submissions"
        
        # Count submissions within the time window
        window_start = current_time - timedelta(minutes=time_window_minutes)
        recent_submissions = [
            s for s in user.submissions 
            if hasattr(s, 'timestamp') and s.timestamp >= window_start
        ]
        
        submission_count = len(recent_submissions)
        
        # Define anomaly thresholds
        if time_window_minutes == 60:  # Hourly check
            anomaly_threshold = 10  # More than 10 submissions per hour
        elif time_window_minutes == 10:  # 10-minute check
            anomaly_threshold = 3   # More than 3 submissions in 10 minutes
        else:
            anomaly_threshold = 5   # Default threshold
        
        if submission_count > anomaly_threshold:
            return True, submission_count, f"High frequency detected: {submission_count} submissions in {time_window_minutes} minutes"
        
        return False, submission_count, f"Normal activity: {submission_count} submissions in {time_window_minutes} minutes"


# Singleton instance
ai_service = {
    'recommendations': RecommendationEngine(),
    'verification': VerificationSystem(),
    'tips': EcoTipGenerator(),
    'fraud': FraudDetector(),
    'journal': JournalAnalyzer(),
    'auto_validator': AutoActionValidator(),
    'smart_nudges': SmartNudgeEngine(),
    'civic_explainer': CivicScoreExplainer(),
    'challenge_generator': PersonalizedChallengeGenerator()
}
