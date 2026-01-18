# 🌱 EcoEdu
### AI-Powered Environmental Awareness Platform

EcoEdu is an AI-driven, gamified environmental education platform that motivates students to take real-world eco-friendly actions. By combining Agentic AI, computer vision, recommendation systems, and gamification, EcoEdu transforms sustainability learning into an engaging, measurable, and rewarding experience.

This project was built for a hackathon environment, focusing on innovation, automation, and real-world impact.

## 🌍 Problem Statement

Environmental awareness among students is often limited to theory, with little motivation or verification of real-world action. Existing platforms lack:
*   Action-based learning
*   Automated validation
*   Personalized engagement
*   Formal recognition for impact

EcoEdu solves this gap by encouraging students to perform real eco-activities, verifying them using AI, and rewarding genuine impact with certificates and achievements.

## 🚀 Key Features

### 🧠 Agentic AI Core
EcoEdu uses multiple AI agents working together to automate decisions and improve engagement:

*   **Automated Challenge Verification**
    *   Uses MobileNetV2 (TensorFlow) to analyze uploaded images
    *   Detects eco-activities like tree planting, waste segregation, recycling
    *   Reduces manual verification effort

*   **Smart Challenge Recommendations**
    *   TF-IDF + Cosine Similarity
    *   Suggests challenges based on past behavior and interests

*   **Fraud Detection Agent**
    *   Detects abnormal submission patterns
    *   Prevents point abuse and duplicate uploads

*   **Intelligent Eco-Tips**
    *   Context-aware daily tips for sustained engagement

*   **NLP Journal Analysis**
    *   Uses sentiment analysis (TextBlob)
    *   Encourages students with AI-generated feedback

### 🎮 Gamification & Rewards
*   **Eco-Points & XP System**
    *   Points awarded based on challenge difficulty
    *   Progressive leveling system
*   **Levels**
    1.  Eco Newbie 🌱
    2.  Green Explorer 🍀
    3.  Eco Warrior 🌍
    4.  Planet Protector 🌎
    5.  Earth Guardian 🌳
*   **Achievements & Badges**
    *   Automatically unlocked based on milestones

### 🏆 Automated Certificate System (NEW)
*   Certificates awarded every 1000 Eco-Points
*   Uses a professional PDF template
*   Automatically replaces "NAME" with the student’s username
*   One certificate per milestone (no duplicates)
*   Downloadable from the student dashboard
*   Acts as formal recognition for real-world environmental impact

### 📊 Interactive Dashboard
*   Eco-points growth charts
*   Achievement and challenge analytics
*   Learning journey visualization
*   GitHub-style contribution heatmap
*   Certificate download section

## 👥 User Roles

### 🎓 Student
*   Complete eco-challenges
*   Upload proof images
*   Earn points, badges, levels
*   Receive certificates
*   Track progress via dashboard

### 👨‍🏫 Teacher
*   Verify flagged challenges
*   Monitor student progress
*   Search and filter student data

### 🛠️ Admin
*   Manage users, challenges, achievements
*   Add eco-tips
*   View platform analytics

## 🛠️ Technology Stack

### Backend
*   Python, Flask
*   SQLAlchemy (SQLite / PostgreSQL)

### Frontend
*   HTML5, TailwindCSS
*   Jinja2 Templates
*   Chart.js / Google Charts

### AI / ML
*   TensorFlow (MobileNetV2)
*   Scikit-learn (TF-IDF)
*   TextBlob (NLP)
*   Pillow (Image & Certificate Processing)

### Security
*   Flask-Login
*   Password hashing
*   CSRF protection
*   Rate limiting

## 📦 Installation & Setup

1.  **Clone Repository**
    ```bash
    git clone https://github.com/codinggujarat/EcoEdu.git
    cd EcoEdu
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Application**
    ```bash
    python app.py
    ```

Access at: 👉 http://127.0.0.1:5000

## 🧪 How It Works (Flow)
1.  Student completes a real eco-activity
2.  Uploads image proof
3.  AI verifies activity
4.  Eco-points awarded
5.  Level & achievement updated
6.  Certificate generated automatically at milestones
7.  Progress visible in dashboard

## 📂 Project Structure
```text
EcoEdu/
├── app.py                  # 🚀 Main Entry Point (Flask)
├── ai_service.py           # 🧠 AI Core (Verification, Recommendations, Fraud)
├── certificate_service.py  # 📜 Certificate Generation Engine (Pillow)
├── ml_models.py            # 🤖 ML Model Loader (MobileNetV2)
├── fix_dashboard.py        # 🔧 Utilities
├── requirements.txt        # 📦 Dependencies
├── static/
│   ├── css/                # Style Definitions
│   ├── js/                 # Client-side Logic
│   ├── uploads/            # 🖼️ User Challenge Evidence
│   └── certificates/       # 🎓 Generated PDF Certificates
└── templates/              # 🎨 Jinja2 Templates
    ├── index.html          # Landing Page
    ├── student_dashboard.html # 📊 Main Dashboard
    ├── teacher_dashboard.html # 👨‍🏫 Admin Panel
    ├── login.html          # Auth Pages
    └── ...
```

## 🌟 Innovation & Uniqueness
*   Combines Agentic AI + Gamification + Education
*   Real-world action verification using computer vision
*   Automated milestone-based certification
*   Behavior-change focused design
*   Not just learning → doing + proving + rewarding

## ⚠️ Challenges Faced
*   Ensuring AI verification accuracy across image conditions
*   Preventing system abuse and duplicate rewards
*   Balancing automation with manual oversight
*   Secure and scalable certificate generation
*   Maintaining UX with complex features

## 🏆 Hackathon Details

### **Event**
**Google for Developers

### **Team: CODINGGUJARAT**
*   **Team Lead**: Aman Nayak
*   **Members**:
    *   Vinit Patel
    *   Dadhaniya Hiren

## 🤝 Contribution
1.  Fork the repository
2.  Create a feature branch
3.  Commit changes
4.  Open a Pull Request

## 📝 License
This project is licensed under the MIT License.

---

### 🌱 Final Note
*EcoEdu is not just a project — it is a scalable blueprint for AI-driven environmental action.*
