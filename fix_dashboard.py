
import os

new_script = """    <script>
        document.addEventListener("DOMContentLoaded", function () {
            console.log("🚀 Dashboard Script Started (v3 - Safe JSON)");

            if (typeof Chart === "undefined") {
                console.error("❌ Chart.js not loaded");
                return;
            }

            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: "#9CA3AF" } }
                },
                scales: {
                    x: {
                        ticks: { color: "#9CA3AF" },
                        grid: { color: "rgba(156,163,175,0.1)" }
                    },
                    y: {
                        ticks: { color: "#9CA3AF" },
                        grid: { color: "rgba(156,163,175,0.1)" }
                    }
                }
            };

            /* SAFE DATA INJECTION USING TOJSON */
            // Using tojson ensures strings are quoted and special chars are escaped.
            // For numbers, we default to 0.
            const ecoPoints = {{ user.eco_points | default(0) | tojson }};
            const achievementsCount = {{ user.achievements | length | default(0) | tojson }};
            const challengeCompletions = {{ user.challenge_completions | length | default(0) | tojson }};
            const totalChallenges = {{ challenges | length | default(1) | tojson }};
            
            console.log("Stats:", { ecoPoints, achievementsCount, challengeCompletions, totalChallenges });

            /* CHART 1: Eco Points */
            const pointsCanvas = document.getElementById("pointsGrowthChart");
            if (pointsCanvas) {
                new Chart(pointsCanvas, {
                    type: "line",
                    data: {
                        labels: ["Week 1","Week 2","Week 3","Week 4","Week 5","Current"],
                        datasets: [{
                            label: "Eco Points",
                            data: [50, 120, 200, 350, 480, ecoPoints],
                            borderColor: "#22c55e",
                            backgroundColor: "rgba(34,197,94,0.15)",
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: chartOptions
                });
            }

            /* CHART 2: Achievements */
            const achievementsCanvas = document.getElementById("achievementsChart");
            if (achievementsCanvas) {
                new Chart(achievementsCanvas, {
                    type: "doughnut",
                    data: {
                        labels: ["Earned","Remaining"],
                        datasets: [{
                            data: [achievementsCount, Math.max(20 - achievementsCount, 0)],
                            backgroundColor: ["#22c55e","#374151"]
                        }]
                    },
                    options: chartOptions
                });
            }

            /* CHART 3: Learning Journey */
            // Using tojson for the whole object structure if possible, 
            // but here we manually construct to keep control.
            const levelData = [
                {% if levels %}
                {% for level in levels %}
                {
                    name: {{ level.name | tojson }},
                    progress: {{ level.current_xp | default(0) | tojson }},
                    total: {{ level.next_level_xp | default(100) | tojson }},
                    completed: {{ level.completed | tojson }}
                }{% if not loop.last %},{% endif %}
                {% endfor %}
                {% endif %}
            ];

            const learningCanvas = document.getElementById("learningJourneyChart");
            if (learningCanvas && levelData.length) {
                new Chart(learningCanvas, {
                    type: "bar",
                    data: {
                        labels: levelData.map(l => l.name),
                        datasets: [{
                            label: "Completion %",
                            data: levelData.map(l => Math.round((l.progress / l.total) * 100)),
                            backgroundColor: levelData.map(l => l.completed ? "#22c55e" : "#16a34a")
                        }]
                    },
                    options: chartOptions
                });
            }

            /* CHART 4: Challenges */
            const challengesCanvas = document.getElementById("challengesChart");
            if (challengesCanvas) {
                new Chart(challengesCanvas, {
                    type: "polarArea",
                    data: {
                        labels: ["Completed","Pending"],
                        datasets: [{
                            data: [challengeCompletions, totalChallenges - challengeCompletions],
                            backgroundColor: [
                                "rgba(34,197,94,0.85)",
                                "rgba(234,179,8,0.85)"
                            ]
                        }]
                    },
                    options: chartOptions
                });
            }

        });
    </script>
"""

file_path = 'c:/Users/91704/Downloads/AuthPortal/templates/student_dashboard.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. REMOVE DEBUG PANEL
# Search for the Debug Panel HTML block and remove it
debug_start_marker = '<!-- DEBUG PANEL (Temporary) -->'
debug_end_marker = '<div id="analytics-section"'

if debug_start_marker in content and debug_end_marker in content:
    start_pos = content.find(debug_start_marker)
    valid_content_start = content.find(debug_end_marker)
    
    # We want to keep valid_content_start (the analytics div)
    # So we remove from start_pos up to valid_content_start
    content = content[:start_pos] + content[valid_content_start:]
    print("Debug Panel Removed.")

# 2. REPLACE SCRIPT
# We will locate the script by the "SAFE DATA" comment or similar, or just replace the block again.
# The previous script started around line 1016 (variable, but we can search).
script_start_marker = '<script>'
script_signature = 'document.addEventListener("DOMContentLoaded",'

# Find the LAST script block that contains the signature
start_indices = [i for i in range(len(content)) if content.startswith(script_start_marker, i)]
target_start = -1
target_end = -1

for idx in start_indices:
    # Check if this script block contains our signature
    # Find closing tag
    close_idx = content.find('</script>', idx)
    if close_idx == -1: continue
    
    block = content[idx:close_idx]
    if script_signature in block:
        target_start = idx
        target_end = close_idx + len('</script>')
        break

if target_start != -1:
    print(f"Found target script block at index {target_start}-{target_end}")
    content = content[:target_start] + new_script + content[target_end:]
    print("Script Block Updated.")
else:
    print("WARNING: Could not locate script block to replace. Appending new script.")
    content += "\n" + new_script

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("File processing complete.")
