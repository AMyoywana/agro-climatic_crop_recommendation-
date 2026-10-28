import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crop Recommendation | Ingcebiso Ngezityalo", page_icon="🌱", layout="centered")

MODEL_FILE = "random_forest_model.pkl"
ENCODER_FILE = "label_encoder.pkl"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

TEXT = {
    "English": {
        "title": "🌱 Crop Recommendation",
        "subtitle": "Enter your soil and environmental conditions to receive a crop recommendation from the trained Random Forest model.",
        "location_heading": "Your farming location",
        "location_label": "Location or village (optional)",
        "location_help": "This identifies the location shown in the result. It is not currently used as a model input.",
        "conditions": "Soil and environmental conditions",
        "n": "Nitrogen (N)", "p": "Phosphorus (P)", "k": "Potassium (K)",
        "temp": "Temperature (°C)", "humidity": "Humidity (%)", "ph": "Soil pH", "rain": "Rainfall (mm)",
        "button": "Recommend a crop", "result": "Recommended crop",
        "for_location": "Location entered", "not_entered": "Not specified",
        "disclaimer": "Important: This is a prototype prediction based on the dataset used to train the model. It is not a guarantee of crop success. Check local agronomic advice, soil-test results and seasonal conditions before planting.",
        "model_missing": "Model files were not found. Place random_forest_model.pkl and label_encoder.pkl in the same folder as app.py.",
        "load_error": "The model files could not be loaded. Check that both files came from the same trained model and label encoder.",
        "prediction_error": "The model could not make a prediction from these inputs. Check that the saved model was trained using the seven expected features.",
        "range_note": "Enter values based on a soil test and reliable local weather information where possible. Rainfall should use the same measurement period as the data used for training.",
        "about": "About this prototype",
        "about_text": "The model uses seven inputs: nitrogen, phosphorus, potassium, temperature, humidity, soil pH and rainfall. The location field is descriptive only; automatic weather-forecast retrieval is not implemented in this version.",
        "language": "Interface language / Ulwimi"
    },
    "isiXhosa": {
        "title": "🌱 Ingcebiso Ngezityalo",
        "subtitle": "Faka iimeko zomhlaba nezendalo ukuze ufumane isindululo sesityalo kwimodeli yeRandom Forest eqeqeshiweyo.",
        "location_heading": "Indawo yakho yokulima",
        "location_label": "Indawo okanye ilali (akuyomfuneko)",
        "location_help": "Le ndawo iboniswa kwisiphumo kuphela. Okwangoku ayisetyenziswa yimodeli ukwenza uqikelelo.",
        "conditions": "Iimeko zomhlaba nezendalo",
        "n": "Initrogen (N)", "p": "Iphosphorus (P)", "k": "Ipotassium (K)",
        "temp": "Iqondo lobushushu (°C)", "humidity": "Ukufuma komoya (%)", "ph": "I-pH yomhlaba", "rain": "Imvula (mm)",
        "button": "Cebisa ngesityalo", "result": "Isityalo esicetyiswayo",
        "for_location": "Indawo efakiweyo", "not_entered": "Ayichazwanga",
        "disclaimer": "Kubalulekile: Esi sisiphumo semodeli esekelwe kwidatha ebisetyenziselwa ukuyiqeqesha. Asiqinisekisi ukuba isityalo siya kuphumelela. Qhagamshelana nengcali yezolimo yasekuhlaleni, ujonge iziphumo zovavanyo lomhlaba neemeko zexesha lonyaka ngaphambi kokutyala.",
        "model_missing": "Iifayile zemodeli azifumanekanga. Beka i-random_forest_model.pkl kunye ne-label_encoder.pkl kwifolda enye ne-app.py.",
        "load_error": "Iifayile zemodeli azikwazanga ukuvulwa. Qinisekisa ukuba zombini zivela kwimodeli eqeqeshiweyo nakwi-label encoder efanayo.",
        "prediction_error": "Imodeli ayikwazanga ukwenza uqikelelo ngezi datha. Qinisekisa ukuba imodeli eqeqeshiweyo isebenzisa ezi mpawu zisixhenxe zilindelekileyo.",
        "range_note": "Sebenzisa amaxabiso ovavanyo lomhlaba nolwazi oluthembekileyo lwemozulu xa kunokwenzeka. Imvula mayibe yeyexesha elifanayo nelasetyenziswa kwidatha yoqeqesho.",
        "about": "Malunga nale nkqubo",
        "about_text": "Imodeli isebenzisa iimpawu ezisixhenxe: initrogen, iphosphorus, ipotassium, ubushushu, ukufuma, i-pH yomhlaba nemvula. Indawo efakiweyo iboniswa kuphela; ukufumana uqikelelo lwemozulu ngokuzenzekelayo akukafakwa kolu guqulelo.",
        "language": "Interface language / Ulwimi"
    }
}

lang = st.sidebar.selectbox("Interface language / Ulwimi", ["English", "isiXhosa"])
t = TEXT[lang]

st.title(t["title"])
st.write(t["subtitle"])

@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_FILE), joblib.load(ENCODER_FILE)

if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODER_FILE):
    st.error(t["model_missing"])
    st.stop()

try:
    model, label_encoder = load_artifacts()
except Exception:
    st.error(t["load_error"])
    st.stop()

st.subheader(t["location_heading"])
location = st.text_input(
    t["location_label"],
    placeholder="e.g., Alice, Eastern Cape / umzekelo, eAlice",
    help=t["location_help"]
)

st.subheader(t["conditions"])
st.caption(t["range_note"])

with st.form("crop_recommendation_form"):
    col1, col2 = st.columns(2)
    with col1:
        nitrogen = st.number_input(t["n"], min_value=0.0, max_value=300.0, value=90.0, step=1.0)
        phosphorus = st.number_input(t["p"], min_value=0.0, max_value=300.0, value=42.0, step=1.0)
        potassium = st.number_input(t["k"], min_value=0.0, max_value=300.0, value=43.0, step=1.0)
        temperature = st.number_input(t["temp"], min_value=-10.0, max_value=60.0, value=25.0, step=0.5)
    with col2:
        humidity = st.number_input(t["humidity"], min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        ph = st.number_input(t["ph"], min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        rainfall = st.number_input(t["rain"], min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
    submitted = st.form_submit_button(t["button"], type="primary", use_container_width=True)

if submitted:
    try:
        input_data = pd.DataFrame(
            [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]],
            columns=FEATURES
        )
        prediction = model.predict(input_data)
        crop_name = label_encoder.inverse_transform(prediction.astype(int))[0]
        st.success(f"### {t['result']}: {str(crop_name).replace('_', ' ').title()}")
        st.write(f"**{t['for_location']}:** {location.strip() if location.strip() else t['not_entered']}")
        st.info(t["disclaimer"])
    except Exception as exc:
        st.error(t["prediction_error"])
        st.caption(f"Technical detail: {exc}")

with st.expander(t["about"]):
    st.write(t["about_text"])

st.caption("Research prototype | University of Fort Hare")
