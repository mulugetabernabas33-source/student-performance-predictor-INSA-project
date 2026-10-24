"""Human-readable labels for the dataset's coded categorical/ordinal values."""

EDUCATION_LEVELS = {
    0: "None",
    1: "Primary (4th grade)",
    2: "5th-9th grade",
    3: "Secondary education",
    4: "Higher education",
}
TRAVEL_TIMES = {1: "Under 15 min", 2: "15-30 min", 3: "30-60 min", 4: "Over 1 hour"}
STUDY_TIMES = {1: "Under 2 hrs/week", 2: "2-5 hrs/week", 3: "5-10 hrs/week", 4: "Over 10 hrs/week"}
FIVE_POINT_SCALE = {1: "Very low", 2: "Low", 3: "Medium", 4: "High", 5: "Very high"}
YES_NO = {"yes": "Yes", "no": "No"}
JOBS = ["at_home", "health", "other", "services", "teacher"]
JOB_LABELS = {
    "at_home": "At home",
    "health": "Healthcare",
    "other": "Other",
    "services": "Civil services",
    "teacher": "Teacher",
}
REASONS = {"course": "Course preference", "home": "Close to home", "reputation": "School reputation", "other": "Other"}
GUARDIANS = {"mother": "Mother", "father": "Father", "other": "Other"}

# Plain-language, actionable advice keyed by raw feature name, shown next to
# SHAP factors that push a prediction toward "at risk". Generic fallback is
# used for any feature not covered here.
RISK_FACTOR_ADVICE = {
    "failures": "Past course failures are the single strongest risk signal in this model. Prioritize this student for tutoring or a structured study plan.",
    "absences": "Frequent absences correlate strongly with lower grades. An attendance check-in is often the highest-leverage early intervention.",
    "studytime": "Low weekly study time is linked to weaker outcomes. Encourage a concrete, scheduled study routine rather than open-ended revision.",
    "goout": "Frequent socializing is associated with less study time. Not inherently a problem, but worth balancing against the study-time signal.",
    "Dalc": "Elevated workday alcohol consumption is associated with lower academic performance; this may warrant a wellbeing conversation.",
    "Walc": "Elevated weekend alcohol consumption shows a mild association with lower grades.",
    "schoolsup": "No extra educational support is currently in place; enrolling in school-provided support may help.",
    "famsup": "Limited family educational support at home can be offset with school-based tutoring or study groups.",
    "traveltime": "A long commute leaves less time and energy for study; consider whether school-based study time could help.",
    "freetime": "Very high unstructured free time, without a study routine, is weakly associated with lower grades.",
    "health": "Lower reported health status can affect concentration and attendance; a wellbeing check-in may be worthwhile.",
    "age": "Older students (relative to peers in the same grade) show a mild negative association with final grades, often linked to prior repeated years.",
    "romantic": "Being in a relationship shows a small negative association with grades in this dataset, likely via reduced study time.",
    "internet": "No internet access at home can limit access to research and practice resources outside school hours.",
    "higher": "Lower stated intent to pursue higher education is one of the strongest motivational signals in this dataset.",
}
GENERIC_ADVICE = "This factor contributes to the risk estimate; combine it with teacher judgment rather than acting on it alone."
