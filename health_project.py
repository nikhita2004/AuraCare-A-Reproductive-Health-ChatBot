import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import gradio as gr
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# Step 1: Data Preparation
np.random.seed(42)
n_ladies = 5000
age = np.random.normal(30, 8, n_ladies)
bmi = np.random.normal(25, 5, n_ladies) + 0.1 * age / 10
smoking = np.random.binomial(1, 0.2, n_ladies)
family_history_pcod = np.random.binomial(1, 0.25, n_ladies)
exercise_hours = np.random.normal(2, 1, n_ladies).clip(0, 5)
hpv_test = np.random.binomial(1, 0.15 + 0.05 * smoking + 0.05 * (age > 30), n_ladies)
biopsy_result = np.random.binomial(1, 0.05 + 0.1 * hpv_test + 0.05 * (age > 40), n_ladies)
pregnancy_stage = np.random.choice(['Pre', 'Pregnant', 'Post'], n_ladies, p=[0.4, 0.3, 0.3])
blood_pressure = np.random.normal(120, 15, n_ladies) + 10 * (pregnancy_stage == 'Pregnant')
blood_sugar = np.random.normal(8, 2, n_ladies) + 2 * (pregnancy_stage == 'Pregnant') + 0.5 * (bmi > 30)
irregular_periods = np.random.binomial(1, 0.3 + 0.2 * (bmi > 30) + 0.15 * family_history_pcod, n_ladies)
hormone_ratio = np.random.normal(1.5, 0.5, n_ladies) + 0.5 * irregular_periods + 0.3 * (bmi > 30)
cycle_length = np.clip(np.random.normal(28, 4, n_ladies), 21, 35)
pain_level = np.random.randint(0, 6, n_ladies) + irregular_periods
postpartum_depression = np.random.normal(3, 2, n_ladies).clip(0, 10) + 2 * (pregnancy_stage == 'Post') + 1 * (pain_level > 3)
recovery_time = np.random.normal(6, 2, n_ladies).clip(0, 12) + 1 * (postpartum_depression > 5)
prenatal_visits = np.random.poisson(5, n_ladies)
postnatal_checkups = np.random.poisson(3, n_ladies)

data_dict = {
    'Age': age, 'BMI': bmi, 'Smoking': smoking, 'Family_History_PCOD': family_history_pcod,
    'Exercise_Hours': exercise_hours, 'HPV_Test': hpv_test, 'Biopsy_Result': biopsy_result,
    'Pregnancy_Stage': pregnancy_stage, 'BloodPressure': blood_pressure, 'BloodSugar': blood_sugar,
    'Irregular_Periods': irregular_periods, 'Hormone_Ratio': hormone_ratio, 'Cycle_Length': cycle_length,
    'Pain_Level': pain_level, 'Postpartum_Depression': postpartum_depression, 'Recovery_Time': recovery_time,
    'Prenatal_Visits': prenatal_visits, 'Postnatal_Checkups': postnatal_checkups
}
synthetic_df = pd.DataFrame(data_dict)
synthetic_df['Health_Risk_Score'] = (
    (synthetic_df['Biopsy_Result'] * 3) +
    (synthetic_df['BloodPressure'] / 40) +
    (synthetic_df['BloodSugar'] / 2) +
    (synthetic_df['Postpartum_Depression'] / 2) +
    (synthetic_df['Irregular_Periods'] * 1) +
    (synthetic_df['Pain_Level'] / 2) +
    (synthetic_df['Family_History_PCOD'] * 0.5) -
    (synthetic_df['Exercise_Hours'] * 0.2)
).clip(0, 10)

# Step 2: Exploratory Data Analysis
print("Let’s check the numbers!")
print(synthetic_df.describe())
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='BMI', hue='Pregnancy_Stage', size='Health_Risk_Score', data=synthetic_df)
plt.title('Age vs BMI by Pregnancy Stage')
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pregnancy_Stage', y='BloodPressure', data=synthetic_df)
plt.title('Blood Pressure by Pregnancy Stage')
plt.show()
plt.figure(figsize=(10, 6))
sns.histplot(data=synthetic_df, x='Exercise_Hours', hue='Irregular_Periods', multiple="stack")
plt.title('Exercise Hours vs Irregular Periods')
plt.show()

# Step 3: Modeling
features = ['Age', 'BMI', 'Smoking', 'Family_History_PCOD', 'Exercise_Hours', 'HPV_Test', 'Biopsy_Result',
            'Pregnancy_Stage', 'BloodPressure', 'BloodSugar', 'Irregular_Periods', 'Hormone_Ratio',
            'Cycle_Length', 'Pain_Level', 'Postpartum_Depression', 'Recovery_Time', 'Prenatal_Visits',
            'Postnatal_Checkups']
X_synthetic = synthetic_df[features]
X_synthetic = pd.get_dummies(X_synthetic, columns=['Pregnancy_Stage'], drop_first=True)
scaler = StandardScaler()
X_scaled_synthetic = scaler.fit_transform(X_synthetic)
n_groups = 4
gmm = GaussianMixture(n_components=n_groups, random_state=42)
synthetic_df['GMM_Cluster'] = gmm.fit_predict(X_scaled_synthetic)

numeric_cols = synthetic_df.select_dtypes(include=np.number).columns.tolist()
group_info = synthetic_df.groupby('GMM_Cluster')[numeric_cols].mean()

def give_advice(group_num, health_score, lady_info):
    advice_list = []
    confidence = 0.7
    group_avg = group_info.loc[group_num]

    if (lady_info.get('Biopsy_Result', 0) == 1 or lady_info.get('HPV_Test', 0) == 1 or 
        lady_info.get('Smoking', 0) == 1 or group_avg['Biopsy_Result'] > 0.1):
        advice_list.append("Go see a doctor for a cervical cancer check!")
        confidence = max(confidence, 0.85)

    if (lady_info.get('BloodPressure', 0) > 130 or lady_info.get('BloodSugar', 0) > 9 or 
        lady_info.get('Pregnancy_Stage', 'Pre') == 'Pregnant' or 
        (group_avg['BloodPressure'] > 125 and lady_info.get('Pregnancy_Stage', 'Pre') in ['Pregnant', 'Post'])):
        advice_list.append("Visit an obstetrician to check your pregnancy!")
        confidence = max(confidence, 0.8)

    if (lady_info.get('Irregular_Periods', 0) == 1 or lady_info.get('Hormone_Ratio', 0) > 2.0 or 
        lady_info.get('Family_History_PCOD', 0) == 1 or group_avg['Irregular_Periods'] > 0.4):
        advice_list.append("Talk to a gynecologist about PCOD!")
        confidence = max(confidence, 0.75)

    if (lady_info.get('Postpartum_Depression', 0) > 5 or lady_info.get('Recovery_Time', 0) > 8 or 
        lady_info.get('Pregnancy_Stage', 'Pre') == 'Post'):
        advice_list.append("Get help for postpartum feelings from a counselor!")
        confidence = max(confidence, 0.8)

    if (lady_info.get('Cycle_Length', 28) < 24 or lady_info.get('Cycle_Length', 28) > 32 or 
        lady_info.get('Pain_Level', 0) > 4 or group_avg['Pain_Level'] > 3):
        advice_list.append("Check your periods with a doctor if they hurt a lot!")
        confidence = max(confidence, 0.7)

    if health_score > 7:
        advice_list.append("Oh no! High risk! Get a full checkup NOW!")
        confidence = 0.9
    elif health_score > 4:
        advice_list.append("Be careful! Keep checking your health regularly.")
        confidence = 0.65

    return advice_list, confidence if advice_list else (["You’re doing great! Keep up checkups!"], 0.6)

# Step 4: Evaluation
synthetic_df['True_Risk'] = np.where(synthetic_df['Health_Risk_Score'] > 6, 1, 0)
synthetic_df['Advice'], synthetic_df['Confidence'] = zip(*synthetic_df.apply(
    lambda row: give_advice(row['GMM_Cluster'], row['Health_Risk_Score'], row), axis=1
))
def guess_risk_from_advice(advice):
    return 1 if "Oh no!" in '; '.join(advice) else 0
synthetic_df['Guessed_Risk'] = synthetic_df['Advice'].apply(guess_risk_from_advice)
accuracy = accuracy_score(synthetic_df['True_Risk'], synthetic_df['Guessed_Risk'])
print(f"My accuracy is: {accuracy:.2%}")
print(f"Average Confidence: {synthetic_df['Confidence'].mean():.2f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_synthetic)
synthetic_df['PCA1'] = X_pca[:, 0]
synthetic_df['PCA2'] = X_pca[:, 1]
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='GMM_Cluster', size='Health_Risk_Score', data=synthetic_df, palette='rainbow')
plt.title('My Amazing Groups!')
plt.show()

# Step 5: Extra Features - Save to File and PDF
def save_advice(lady_info, advice, confidence, filename="health_advice.txt"):
    with open(filename, "a") as file:
        file.write(f"\nDate: {pd.Timestamp.now()}\n")
        file.write(f"User Info: {lady_info}\n")
        file.write(f"Advice: {advice}\n")
        file.write(f"Confidence: {confidence:.2f}\n")
        file.write("-" * 50 + "\n")
    return "Advice saved to file!"

def save_to_pdf(lady_info, advice, confidence, filename="health_advice.pdf"):
    pdf = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"<b>Reproductive Health Report</b><br/>Date: {pd.Timestamp.now()}", styles['Heading1']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>User Info:</b> {lady_info}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Advice:</b> {', '.join(advice)}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2%}", styles['Normal']))
    story.append(Spacer(1, 12))
    if os.path.exists('health_comparison.png'):
        story.append(Image('health_comparison.png', width=200, height=150))
    pdf.build(story)
    return f"PDF saved as {filename}!"

# Step 6: Interactive Chatbot - Define questions without age (already asked)
teen_questions = [
    ("What’s your BMI?", "¿Cuál es tu IMC?", "आपका बीएमआई क्या है? (Aapka BMI kya hai?)", "నీ బీఎంఐ ఎంత ఉంది? (Nī BMĪ enta undi?)", "BMI", "number"),
    ("Do you have irregular periods? (Yes/No)", "¿Tienes períodos irregulares? (Sí/No)", "क्या आपके मासिक धर्म अनियमित हैं? (हाँ/नहीं) (Kya aapke masik dharm aniyamit hain? (Haan/Nahi))", "నీకు అసమాన కాలాలు ఉన్నాయా? (అవును/కాదు) (Nīku asamāna kālālu unnāyā? (Avunu/Kādu))", "Irregular_Periods", "text"),
    ("Do you have a family history of PCOD? (Yes/No)", "¿Tienes antecedentes familiares de PCOD? (Sí/No)", "क्या आपके परिवार में PCOD का इतिहास है? (हाँ/नहीं) (Kya aapke parivar mein PCOD ka itihas hai? (Haan/Nahi))", "నీ కుటుంబంలో PCOD చరిత్ర ఉందా? (అవును/కాదు) (Nī kuṭumbam̐lō PCOD charitra undā? (Avunu/Kādu))", "Family_History_PCOD", "text"),
    ("How much pain do you feel during periods (0-5)?", "¿Cuánto dolor sientes durante tus períodos (0-5)?", "मासिक धर्म के दौरान आपको कितना दर्द होता है (0-5)? (Masik dharm ke dauraan aapko kitna dard hota hai (0-5)?)", "పీరియడ్స్ సమయంలో నీకు ఎంత నొప్పి ఉంది (0-5)? (Pīriyarḍs samayaṁlō nīku enta noppi undi (0-5)?)", "Pain_Level", "number"),
    ("How long is your menstrual cycle (in days)?", "¿Cuánto dura tu ciclo menstrual (en días)?", "आपका मासिक चक्र कितने दिनों का है? (Aapka masik chakra kitne dinon ka hai?)", "నీ మాసిక చక్రం ఎన్ని రోజులు ఉంటుంది? (Nī māsika chakram enni rojulu uṇṭundi?)", "Cycle_Length", "number"),
    ("How many hours do you exercise weekly (0-5)?", "¿Cuántas horas haces ejercicio a la semana (0-5)?", "आप हफ्ते में कितने घंटे व्यायाम करते हैं (0-5)? (Aap hafte mein kitne ghante vyayam karte hain (0-5)?)", "నీవు వారానికి ఎన్ని గంటలు వ్యాయామం చేస్తావు (0-5)? (Nīvu vārāniki enni gaṇṭalu vyāyāmam chestāvu (0-5)?)", "Exercise_Hours", "number")
]

adult_questions = [
    ("What’s your BMI?", "¿Cuál es tu IMC?", "आपका बीएमआई क्या है? (Aapka BMI kya hai?)", "నీ బీఎంఐ ఎంత ఉంది? (Nī BMĪ enta undi?)", "BMI", "number"),
    ("Do you smoke? (Yes/No)", "¿Fumas? (Sí/No)", "क्या आप धूम्रपान करते हैं? (हाँ/नहीं) (Kya aap dhoomrapaan karte hain? (Haan/Nahi))", "నీవు ధూమపానం చేస్తావా? (అవును/కాదు) (Nīvu dhūmapānam chestāvā? (Avunu/Kādu))", "Smoking", "text"),
    ("Have you tested positive for HPV? (Yes/No)", "¿Has dado positivo en la prueba de VPH? (Sí/No)", "क्या आपकी HPV जांच पॉजिटिव आई है? (हाँ/नहीं) (Kya aapki HPV jaanch positive aayi hai? (Haan/Nahi))", "నీకు HPV పరీక్ష పాజిటివ్ వచ్చిందా? (అవును/కాదు) (Nīku HPV parīkṣa pājīṭiv vaccindā? (Avunu/Kādu))", "HPV_Test", "text"),
    ("Is your biopsy result positive? (Yes/No)", "¿Es positivo el resultado de tu biopsia? (Sí/No)", "क्या आपकी बायोप्सी का परिणाम पॉजिटिव है? (हाँ/नहीं) (Kya aapki biopsy ka parinaam positive hai? (Haan/Nahi))", "నీ బయాప్సీ ఫలితం పాజిటివ్ అయినదా? (అవును/కాదు) (Nī bayāpsi phalitam pājīṭiv ayinadā? (Avunu/Kādu))", "Biopsy_Result", "text"),
    ("Are you Pre-pregnant, Pregnant, or Postpartum? (Type 'Pre', 'Pregnant', or 'Post')", "¿Estás Pre-embarazada, Embarazada o Posparto? (Escribe 'Pre', 'Embarazada', o 'Post')", "क्या आप गर्भावस्था से पहले, गर्भवती, या प्रसवोत्तर हैं? ('Pre', 'Pregnant', या 'Post' टाइप करें) (Kya aap garbhavastha se pehle, garbhvati, ya prasavottar hain? ('Pre', 'Pregnant', ya 'Post' type karein))", "నీవు ప్రీ-గర్భవతి, గర్భవతి, లేదా పోస్ట్‌పార్టమ్‌ను? ('Pre', 'Pregnant', లేదా 'Post' ని టైప్ చేయండి) (Nīvu prī-garbhavati, garbhavati, lēdā pōsṭpārtamnu? ('Pre', 'Pregnant', lēdā 'Post' ni ṭaip cēyaṇḍi))", "Pregnancy_Stage", "text"),
    ("What’s your blood pressure?", "¿Cuál es tu presión arterial?", "आपका रक्तचाप क्या है? (Aapka raktachaap kya hai?)", "నీ రక్తపోటు ఎంత ఉంది? (Nī raktapōṭu enta undi?)", "BloodPressure", "number"),
    ("What’s your blood sugar level?", "¿Cuál es tu nivel de azúcar en sangre?", "आपका रक्त शर्करा स्तर क्या है? (Aapka rakt sharkara star kya hai?)", "నీ రక్తంలో చక్కెర స్థాయి ఎంత? (Nī raktam̐lō cakker sthāyi enta?)", "BloodSugar", "number"),
    ("Do you have irregular periods? (Yes/No)", "¿Tienes períodos irregulares? (Sí/No)", "क्या आपके मासिक धर्म अनियमित हैं? (हाँ/नहीं) (Kya aapke masik dharm aniyamit hain? (Haan/Nahi))", "నీకు అసమాన కాలాలు ఉన్నాయా? (అవును/కాదు) (Nīku asamāna kālālu unnāyā? (Avunu/Kādu))", "Irregular_Periods", "text"),
    ("How would you rate your postpartum depression (0-10)?", "¿Cómo calificarías tu depresión posparto (0-10)?", "आप अपनी प्रसवोत्तर अवसाद को 0-10 में कैसे रेट करेंगे? (Aap apni prasavottar avasad ko 0-10 mein kaise rate karenge?)", "నీ పోస్ట్‌పార్టమ్ డిప్రెషన్‌ను 0-10లో ఎలా రేట్ చేస్తావు? (Nī pōsṭpārtam ḍipreśannu 0-10lō elā rēṭ chestāvu?)", "Postpartum_Depression", "number"),
    ("How many hours do you exercise weekly (0-5)?", "¿Cuántas horas haces ejercicio a la semana (0-5)?", "आप हफ्ते में कितने घंटे व्यायाम करते हैं (0-5)? (Aap hafte mein kitne ghante vyayam karte hain (0-5)?)", "నీవు వారానికి ఎన్ని గంటలు వ్యాయామం చేస్తావు (0-5)? (Nīvu vārāniki enni gaṇṭalu vyāyāmam chestāvu (0-5)?)", "Exercise_Hours", "number")
]

postmenopause_questions = [
    ("What’s your BMI?", "¿Cuál es tu IMC?", "आपका बीएमआई क्या है? (Aapka BMI kya hai?)", "నీ బీఎంఐ ఎంత ఉంది? (Nī BMĪ enta undi?)", "BMI", "number"),
    ("Do you smoke? (Yes/No)", "¿Fumas? (Sí/No)", "क्या आप धूम्रपान करते हैं? (हाँ/नहीं) (Kya aap dhoomrapaan karte hain? (Haan/Nahi))", "నీవు ధూమపానం చేస్తావా? (అవును/కాదు) (Nīvu dhūmapānam chestāvā? (Avunu/Kādu))", "Smoking", "text"),
    ("Have you tested positive for HPV? (Yes/No)", "¿Has dado positivo en la prueba de VPH? (Sí/No)", "क्या आपकी HPV जांच पॉजिटివ आई है? (हाँ/नहीं) (Kya aapki HPV jaanch positive aayi hai? (Haan/Nahi))", "నీకు HPV పరీక్ష పాజిటివ్ వచ్చిందా? (అవును/కాదు) (Nīku HPV parīkṣa pājīṭiv vaccindā? (Avunu/Kādu))", "HPV_Test", "text"),
    ("Is your biopsy result positive? (Yes/No)", "¿Es positivo el resultado de tu biopsia? (Sí/No)", "क्या आपकी बायोप्सी का परिणाम पॉजिटिव है? (हाँ/नहीं) (Kya aapki biopsy ka parinaam positive hai? (Haan/Nahi))", "నీ బయాప్సీ ఫలితం పాజిటివ్ అయినదా? (అవును/కాదు) (Nī bayāpsi phalitam pājīṭiv ayinadā? (Avunu/Kādu))", "Biopsy_Result", "text"),
    ("What’s your blood pressure?", "¿Cuál es tu presión arterial?", "आपका रक्तचाप क्या है? (Aapka raktachaap kya hai?)", "నీ రక్తపోటు ఎంత ఉంది? (Nī raktapōṭu enta undi?)", "BloodPressure", "number"),
    ("What’s your blood sugar level?", "¿Cuál es tu nivel de azúcar en sangre?", "आपका रक्त शर्करा स्तर क्या है? (Aapka rakt sharkara star kya hai?)", "నీ రక్తంలో చక్కెర స్థాయి ఎంత? (Nī raktam̐lō cakker sthāyi enta?)", "BloodSugar", "number"),
    ("How many hours do you exercise weekly (0-5)?", "¿Cuántas horas haces ejercicio a la semana (0-5)?", "आप हफ्ते में कितने घंटे व्यायाम करते हैं (0-5)? (Aap hafte mein kitne ghante vyayam karte hain (0-5)?)", "నీవు వారానికి ఎన్ని గంటలు వ్యాయామం చేస్తావు (0-5)? (Nīvu vārāniki enni gaṇṭalu vyāyāmam chestāvu (0-5)?)", "Exercise_Hours", "number")
]

# Translation Dictionary for Messages
translations = {
    "English": {
        "welcome": "🌸 Welcome to the Reproductive Health Chatbot! I’m your Health Buddy, here to guide you with personalized advice. To get started, please enter your age (e.g., 15, 30, or 55) in the box below and press Enter or Submit!",
        "welcome_restart": "Great! Let’s start over.\n🌸 Welcome back! Please enter your age (e.g., 15, 30, or 55) to begin!",
        "error_age": "Oops! Please enter a valid age (e.g., 15, 30, or 55). What’s your age?",
        "error_input": "Oops! Please enter a valid answer for '{question}'. Let’s try again.\n{question}",
        "advice": "Thanks for sharing! You’re in Group {group} with a health score of {health_score:.2f}.\nHere’s my advice for you: {advice}\nConfidence in this advice: {confidence:.2%}\nWould you like to start over? Type 'Yes' to try again!\nOr click the 'Save as PDF' button below to download your report!",
        "chart_message": "Here’s a comparison of your health metrics with your group average:"
    },
    "Spanish": {
        "welcome": "🌸 ¡Bienvenido al Chatbot de Salud Reproductiva! Soy tu Asistente de Salud, aquí para guiarte con consejos personalizados. ¡Para comenzar, por favor ingresa tu edad (ej. 15, 30 o 55) en el cuadro a continuación y presiona Enter o Enviar!",
        "welcome_restart": "¡Genial! Empecemos de nuevo.\n🌸 ¡Bienvenido de vuelta! Por favor, ingresa tu edad (ej. 15, 30 o 55) para comenzar!",
        "error_age": "¡Ups! Por favor, ingresa una edad válida (ej. 15, 30 o 55). ¿Cuál es tu edad?",
        "error_input": "¡Ups! Por favor, ingresa una respuesta válida para '{question}'. Intentémoslo de nuevo.\n{question}",
        "advice": "¡Gracias por compartir! Estás en el Grupo {group} con una puntuación de salud de {health_score:.2f}.\nAquí tienes mi consejo para ti: {advice}\nConfianza en este consejo: {confidence:.2%}\n¿Te gustaría empezar de nuevo? ¡Escribe 'Sí' para intentarlo otra vez!\nO haz clic en el botón 'Guardar como PDF' para descargar tu informe.",
        "chart_message": "Aquí tienes una comparación de tus métricas de salud con el promedio de tu grupo:"
    },
    "Hindi": {
        "welcome": "🌸 प्रजनन स्वास्थ्य चैटबॉट में आपका स्वागत है! मैं आपका स्वास्थ्य सहायक हूँ, जो आपको व्यक्तिगत सलाह देने के लिए यहाँ हूँ। शुरू करने के लिए, कृपया नीचे दिए गए बॉक्स में अपनी उम्र (जैसे, 15, 30, या 55) दर्ज करें और Enter या Submit दबाएँ! (Prajanan Swasthya Chatbot mein aapka swagat hai! Main aapka swasthya sahayak hoon, jo aapko vyaktigat salah dene ke liye yahan hoon. Shuru karne ke liye, kripya neeche diye gaye box mein apni umar (jaise, 15, 30, ya 55) darj karein aur Enter ya Submit dabayein!)",
        "welcome_restart": "शानदार! आइए फिर से शुरू करें।\n🌸 वापस स्वागत है! कृपया अपनी उम्र (जैसे, 15, 30, या 55) दर्ज करें शुरू करने के लिए! (Shandaar! Aaiye phir se shuru karein.\nVapas swagat hai! Kripya apni umar (jaise, 15, 30, ya 55) darj karein shuru karne ke liye!)",
        "error_age": "उफ़! कृपया एक मान्य उम्र दर्ज करें (जैसे, 15, 30, या 55)। आपकी उम्र क्या है? (Oof! Kripya ek manya umar darj karein (jaise, 15, 30, ya 55). Aapki umar kya hai?)",
        "error_input": "उफ़! कृपया '{question}' के लिए एक मान्य उत्तर दर्ज करें। आइए फिर से कोशिश करें।\n{question} (Oof! Kripya '{question}' ke liye ek manya uttar darj karein. Aaiye phir se koshish karein.\n{question})",
        "advice": "साझा करने के लिए धन्यवाद! आप समूह {group} में हैं और आपका स्वास्थ्य स्कोर {health_score:.2f} है।\nयहाँ आपके लिए मेरी सलाह है: {advice}\nइस सलाह में विश्वास: {confidence:.2%}\nक्या आप फिर से शुरू करना चाहेंगे? 'हाँ' टाइप करें फिर से कोशिश करने के लिए!\nया नीचे 'रिपोर्ट PDF के रूप में डाउनलोड करें' बटन पर क्लिक करें अपने रिपोर्ट को सहेजने के लिए! (Sajha karne ke liye dhanyavaad! Aap samuh {group} mein hain aur aapka swasthya score {health_score:.2f} hai.\nYahan aapke liye meri salah hai: {advice}\nIs salah mein vishwas: {confidence:.2%}\nKya aap phir se shuru karna chahenge? 'Haan' type karein phir se koshish karne ke liye!\nYa neeche 'Report PDF ke roop mein download karein' button par click karein apne report ko sahejne ke liye!)",
        "chart_message": "यहाँ आपके स्वास्थ्य मेट्रिक्स की आपके समूह औसत के साथ तुलना है: (Yahan aapke swasthya metrics ki aapke samuh ausat ke saath tulna hai:)"
    },
    "Telugu": {
        "welcome": "🌸 పునరుత్పత్తి ఆరోగ్య చాట్‌బాట్‌కు స్వాగతం! నేను నీ ఆరోగ్య సహాయకుడిని, నిన్ను వ్యక్తిగత సలహాలతో మార్గదర్శకం చేస్తాను. మొదలుపెట్టడానికి, దయచేసి కింది బాక్స్‌లో నీ వయస్సు (ఉదా. 15, 30, లేదా 55) ని నమోదు చేయండి మరియు Enter లేదా Submit ను నొక్కండి! (Punurutpatti ārōgya chāṭbaṭku svāgatam! Nēnu nī ārōgya sahāyakudi, ninna vyaktigata salahālato mārgadarśakam cēstānu. Modalupeṭṭaḍāniki, dayachēsi kin̄di bāks lō nī vayassu (udā. 15, 30, lēdā 55) ni namodu cēyaṇḍi mariyu Enter lēdā Submit nu nokkanḍi!)",
        "welcome_restart": "గొప్ప! మరలా మొదలుపెట్టండి.\n🌸 తిరిగి స్వాగతం! మొదలుపెట్టడానికి దయచేసి నీ వయస్సు (ఉదా. 15, 30, లేదా 55) ని నమోదు చేయండి! (Goppa! Maralā modalupeṭṭanḍi.\nTirigi svāgatam! Modalupeṭṭaḍāniki dayachēsi nī vayassu (udā. 15, 30, lēdā 55) ni namodu cēyaṇḍi!)",
        "error_age": "అయ్యో! దయచేసి ఒక సరైన వయస్సు ని నమోదు చేయండి (ఉదా. 15, 30, లేదా 55). నీ వయస్సు ఎంత? (Ayyō! Dayachēsi oka saraina vayassu ni namodu cēyaṇḍi (udā. 15, 30, lēdā 55). Nī vayassu enta?)",
        "error_input": "అయ్యో! దయచేసి '{question}' కోసం ఒక సరైన సమాధానం నమోదు చేయండి. మరలా ప్రయత్నించండి.\n{question} (Ayyō! Dayachēsi '{question}' kōsam oka saraina samādhānam namodu cēyaṇḍi. Maralā prayatnincanḍi.\n{question})",
        "advice": "షేర్ చేసినందుకు ధన్యవాదాలు! నీవు {group} సమూహంలో ఉన్నావు మరియు నీ ఆరోగ్య స్కోర్ {health_score:.2f} ఉంది.\nనీకు నా సలహా ఇలా ఉంది: {advice}\nఈ సలహా పై నమ్మకం: {confidence:.2%}\nమరలా మొదలుపెట్టాలని అనుకుంటున్నావా? 'అవును' ని టైప్ చేయండి మరలా ప్రయత్నించడానికి!\nలేదా కింది 'PDF గా సేవ్ చేయండి' బటన్‌ను క్లిక్ చేసి నీ నివేదికను డౌన్‌లోడ్ చేయండి! (Śēr cēsinaṇḍuku dhanyavādālū! Nēvu {group} samūham̐lō unnāvu mariyu nī ārōgya skōr {health_score:.2f} undi.\nNīku nā salahā ilā undi: {advice}\nĪ salahā pai nammakam: {confidence:.2%}\nMaralā modalupeṭṭālāni anukuntunnāva? 'Avunu' ni ṭaip cēyaṇḍi maralā prayatnincāḍāniki!\nLēdā kin̄di 'PDF gā sēva cēyaṇḍi' baṭanu klik cēsi nī nivedikanu ḍaunlōḍ cēyaṇḍi!)",
        "chart_message": "ఇక్కడ నీ ఆరోగ్య మెట్రిక్స్‌ను నీ సమూహ సగటుతో పోల్చడం ఉంది: (Ikkada nī ārōgya meṭrikṣnu nī samūha sagaṭutō pōlcaḍam undi:)"
    }
}

def process_answer(user_input, chat_state, chat_history, language):
    lang = language if isinstance(language, str) else "English"
    if "lady_info" not in chat_state:
        chat_state["lady_info"] = {}
        chat_state["current_step"] = 0
        chat_history.append(("Health Buddy", translations[lang]["welcome"]))
        return chat_history, gr.update(value=""), chat_state, gr.update(value=0), None

    current_step = chat_state["current_step"]

    if current_step == 0:
        try:
            age = float(user_input)
            if age < 0 or age > 120:
                raise ValueError("Age should be between 0 and 120.")
            chat_state["lady_info"]["Age"] = age
            if age < 20:
                chat_state["questions"] = teen_questions
            elif age <= 50:
                chat_state["questions"] = adult_questions
            else:
                chat_state["questions"] = postmenopause_questions
            chat_state["current_step"] += 1
            chat_history.append(("User", user_input))
            if not chat_state["questions"]:
                raise ValueError("Questions list is empty after assignment")
            next_question = chat_state["questions"][chat_state["current_step"] - 1][0 if lang == "English" else (1 if lang == "Spanish" else (2 if lang == "Hindi" else 3))]
            chat_history.append(("Health Buddy", next_question))
            progress_value = (chat_state["current_step"] / (len(chat_state["questions"]) + 1)) * 100
            return chat_history, gr.update(value=""), chat_state, gr.update(value=progress_value), None
        except ValueError as e:
            chat_history.append(("Health Buddy", translations[lang]["error_age"]))
            return chat_history, gr.update(value=""), chat_state, gr.update(value=0), None

    question, question_es, question_hi, question_te, key, q_type = chat_state["questions"][current_step - 1]
    question_text = question if lang == "English" else (question_es if lang == "Spanish" else (question_hi if lang == "Hindi" else question_te))

    try:
        if q_type == "number":
            value = float(user_input)
            if key == "BMI" and (value < 10 or value > 50):
                raise ValueError("BMI should be between 10 and 50.")
            elif key == "Exercise_Hours" and (value < 0 or value > 5):
                raise ValueError("Exercise hours should be between 0 and 5.")
            elif key == "Pain_Level" and (value < 0 or value > 5):
                raise ValueError("Pain level should be between 0 and 5.")
            elif key == "Cycle_Length" and (value < 21 or value > 35):
                raise ValueError("Cycle length should be between 21 and 35 days.")
            elif key == "Postpartum_Depression" and (value < 0 or value > 10):
                raise ValueError("Postpartum depression should be between 0 and 10.")
        elif q_type == "text":
            user_input = user_input.strip().lower()
            if key in ['Smoking', 'Family_History_PCOD', 'HPV_Test', 'Biopsy_Result', 'Irregular_Periods']:
                if user_input not in ["yes", "sí", "haan", "అవును", "no"]:
                    raise ValueError("Please answer with 'yes'/'sí'/'haan'/'అవును' or 'no'.")
                value = 1 if user_input in ["yes", "sí", "haan", "అవును"] else 0
            else:
                value = user_input.capitalize()
        chat_state["lady_info"][key] = value
    except ValueError as e:
        chat_history.append(("Health Buddy", translations[lang]["error_input"].format(question=question_text)))
        progress_value = (chat_state["current_step"] / (len(chat_state["questions"]) + 1)) * 100
        return chat_history, gr.update(value=""), chat_state, gr.update(value=progress_value), None

    chat_state["current_step"] += 1
    chat_history.append(("User", user_input))

    if chat_state["current_step"] - 1 >= len(chat_state["questions"]):
        try:
            for feature in features:
                if feature not in chat_state["lady_info"]:
                    if feature in ['Smoking', 'Family_History_PCOD', 'HPV_Test', 'Biopsy_Result', 'Irregular_Periods']:
                        chat_state["lady_info"][feature] = 0
                    elif feature == 'Pregnancy_Stage':
                        chat_state["lady_info"][feature] = 'Pre'
                    elif feature in ['Hormone_Ratio', 'Cycle_Length']:
                        chat_state["lady_info"][feature] = 0.0
                    else:
                        chat_state["lady_info"][feature] = 0.0

            lady_df = pd.DataFrame([chat_state["lady_info"]], columns=features)
            lady_df = pd.get_dummies(lady_df, columns=['Pregnancy_Stage'], drop_first=True)
            for col in X_synthetic.columns:
                if col not in lady_df.columns:
                    lady_df[col] = 0
            lady_df = lady_df[X_synthetic.columns]
            lady_scaled = scaler.transform(lady_df)
            group = gmm.predict(lady_scaled)[0]
            health_score = min(max(
                (chat_state["lady_info"].get('Biopsy_Result', 0) * 3) +
                (chat_state["lady_info"].get('BloodPressure', 0) / 40) +
                (chat_state["lady_info"].get('BloodSugar', 0) / 2) +
                (chat_state["lady_info"].get('Postpartum_Depression', 0) / 2) +
                (chat_state["lady_info"].get('Irregular_Periods', 0) * 1) +
                (chat_state["lady_info"].get('Pain_Level', 0) / 2) +
                (chat_state["lady_info"].get('Family_History_PCOD', 0) * 0.5) -
                (chat_state["lady_info"].get('Exercise_Hours', 0) * 0.2), 0), 10)
            advice, confidence = give_advice(group, health_score, chat_state["lady_info"])
            save_advice(chat_state["lady_info"], advice, confidence)
            
            # Generate Health Metrics Comparison Chart
            group_avg = group_info.loc[group]
            plt.figure(figsize=(6, 4))
            metrics = ['BloodPressure', 'BloodSugar', 'Pain_Level']
            user_values = [chat_state["lady_info"].get(m, 0) for m in metrics]
            avg_values = [group_avg.get(m, 0) for m in metrics]
            x = np.arange(len(metrics))
            plt.bar(x - 0.2, user_values, 0.4, label='Your Values', color='pink')
            plt.bar(x + 0.2, avg_values, 0.4, label='Group Average', color='lightblue')
            plt.xlabel('Health Metrics')
            plt.ylabel('Values')
            plt.title('Your Health Metrics vs Group Average')
            plt.xticks(x, metrics)
            plt.legend()
            plt.tight_layout()
            plt.savefig('health_comparison.png')
            plt.close()
            
            # Update Chat History with Advice and Chart
            response = translations[lang]["advice"].format(group=group, health_score=health_score, advice='\n'.join(advice), confidence=confidence)
            chat_history.append(("Health Buddy", response))
            chat_history.append(("Health Buddy", translations[lang]["chart_message"]))
            chat_history.append((None, gr.Image('health_comparison.png')))
            chat_state["current_step"] = 0
            chat_state["last_advice"] = advice
            chat_state["last_confidence"] = confidence
            return chat_history, gr.update(value=""), chat_state, gr.update(value=100), gr.Image('health_comparison.png')
        except Exception as e:
            chat_history.append(("Health Buddy", f"Oops! Something went wrong: {str(e)}"))
            return chat_history, gr.update(value=""), chat_state, gr.update(value=100), None
    else:
        next_question = chat_state["questions"][chat_state["current_step"] - 1][0 if lang == "English" else (1 if lang == "Spanish" else (2 if lang == "Hindi" else 3))]
        chat_history.append(("Health Buddy", next_question))
        progress_value = (chat_state["current_step"] / (len(chat_state["questions"]) + 1)) * 100
        return chat_history, gr.update(value=""), chat_state, gr.update(value=progress_value), None

def chatbot_response(user_input, chat_history, chat_state, language):
    if not chat_history:
        chat_history = []
    if not chat_state:
        chat_state = {"lady_info": {}, "current_step": 0}

    if user_input.strip().lower() in ["yes", "sí", "haan", "అవును"] and chat_state["current_step"] == 0:
        chat_state["lady_info"] = {}
        chat_state["current_step"] = 0
        lang = language if isinstance(language, str) else "English"
        chat_history = [("Health Buddy", translations[lang]["welcome_restart"])]
        return chat_history, gr.update(value=""), chat_state, gr.update(value=0), None

    return process_answer(user_input, chat_state, chat_history, language)

with gr.Blocks(theme=gr.themes.Soft(primary_hue="pink"), title="🌸AuraCare🌸") as demo:
    gr.Markdown("## 🌸 AuraCare - A Reproductive Health bot 🌸")
    gr.Markdown("Hi! I’m your Health Buddy! I’ll ask questions based on your age and give you personalized advice, one step at a time, Please Enter your age.")
    language = gr.Dropdown(["English", "Spanish", "Hindi", "Telugu"], value="English", label="Language")
    chat_history = gr.Chatbot(label="Chat History", height=400)  # Reverted to default type='tuples'
    user_input = gr.Textbox(label="Your Message", placeholder="Type your answer here...")
    progress = gr.Slider(0, 100, value=0, interactive=False, label="Progress (%)")
    chart_output = gr.Image(label="Health Metrics Comparison")
    chat_state = gr.State(value=None)
    user_input.submit(
        chatbot_response,
        inputs=[user_input, chat_history, chat_state, language],
        outputs=[chat_history, user_input, chat_state, progress, chart_output]
    )
    
    # PDF Download Section
    pdf_button = gr.Button("Save as PDF")
    pdf_output = gr.Textbox(label="PDF Status")
    pdf_button.click(
        fn=lambda state: save_to_pdf(state["lady_info"], state.get("last_advice", []), state.get("last_confidence", 0)),
        inputs=[chat_state],
        outputs=[pdf_output]
    )

demo.launch(server_name="localhost", server_port=7860, share=False)