
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_DIR
    / "Models"
    / "bookwisely_all_models.joblib"
)

ROUTE_PATH = (
    PROJECT_DIR
    / "Data"
    / "processed"
    / "route_stops.parquet"
)


# ==================================================
# 2. CATEGORY DISPLAY NAMES
# ==================================================

CATEGORY_NAMES = {
    "AMTB": "Amrit Bharat Express",
    "DMU": "Diesel Multiple Unit",
    "DRNT": "Duronto Express",
    "EMU": "Electric Multiple Unit",
    "GBR": "Garib Rath",
    "INGL": "Description not verified",
    "JSH": "Jan Shatabdi",
    "MEMU": "Mainline Electric Multiple Unit",
    "MEX": "Mail/Express",
    "PAS": "Passenger",
    "PEXP": "Parcel Express (Non-Passenger)",
    "RAJ": "Rajdhani Express",
    "SHT": "Shatabdi Express",
    "SUB": "Suburban",
    "SUF": "Superfast",
    "TEJ": "Tejas Express",
    "TOD": "Train on Demand",
    "VNDB": "Vande Bharat Express",
    "VNDM": "Namo Bharat Rapid Rail",
    "VNDS": "Vande Bharat Sleeper"
}


def category_label(code):

    description = CATEGORY_NAMES.get(
        code,
        "Description unavailable"
    )

    return f"{code} — {description}"


# ==================================================
# 3. STREAMLIT CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="BookWisely - Train Delay Predictor",
    layout="wide"
)

st.title("BookWisely")
    
st.subheader("Route-Based Train Arrival Delay Predictor")

st.write(
    "Select your source and destination stations to discover "
    "matching train services and predict their arrival delays."
)

st.divider()


# ==================================================
# 4. LOAD SAVED MODELS AND ROUTE DATA
# ==================================================

@st.cache_resource
def load_model_bundle():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_route_data():

    if not ROUTE_PATH.exists():
        raise FileNotFoundError(
            f"Route file not found: {ROUTE_PATH}"
        )

    return pd.read_parquet(ROUTE_PATH)


try:

    bundle = load_model_bundle()
    route_data = load_route_data()

except Exception as error:

    st.error(f"Application data could not be loaded: {error}")
    st.stop()


# Retrieve saved models and preprocessing
models = bundle["models"]

preprocessor = bundle["preprocessor"]

feature_columns = bundle["feature_columns"]


# ==================================================
# 5. PREPARE AVAILABLE STATIONS AND CATEGORIES
# ==================================================

categorical_encoder = (
    preprocessor.named_transformers_["categorical"]
)

# Categories recognized by the trained model
trained_categories = {
    str(value).upper()
    for value in categorical_encoder.categories_[0]
}

# Destination stations recognized by the trained model
trained_stations = {
    str(value).upper()
    for value in categorical_encoder.categories_[1]
}

# Standardize route data
route_data = route_data.copy()

route_data["station_code"] = (
    route_data["station_code"]
    .astype("string")
    .str.strip()
    .str.upper()
)

route_data["category"] = (
    route_data["category"]
    .astype("string")
    .str.strip()
    .str.upper()
)

route_data["train_number"] = (
    route_data["train_number"]
    .astype("string")
    .str.strip()
)

# Exclude parcel services from passenger route search
passenger_routes = route_data.loc[
    route_data["category"] != "PEXP"
].copy()

# Source options can include all passenger-route stations
source_options = sorted(
    passenger_routes["station_code"]
    .dropna()
    .unique()
    .tolist()
)

# Destination must also be recognized by the trained model
destination_options = sorted(
    set(source_options).intersection(trained_stations)
)

# Category options available in the trained model
category_options = sorted(trained_categories)


# ==================================================
# 6. APPLICATION INPUTS
# ==================================================

st.header("Find Your Train Route")

with st.form("route_search_form"):

    col1, col2 = st.columns(2)

    with col1:

        source_station = st.selectbox(
            "Source Station",
            options=source_options,
            index=None,
            placeholder="Type source station code...",
            filter_mode="contains"
        )

    with col2:

        destination_station = st.selectbox(
            "Destination Station",
            options=destination_options,
            index=None,
            placeholder="Type destination station code...",
            filter_mode="contains"
        )

    st.caption(
        "Both station fields support searching with uppercase "
        "or lowercase letters."
    )

    category_choice = st.selectbox(
        "Train Category",
        options=["ALL"] + category_options,
        index=0,
        format_func=lambda code: (
            "All passenger categories"
            if code == "ALL"
            else category_label(code)
        ),
        filter_mode=None
    )

    st.caption(
        "PEXP (Parcel Express) is not included in passenger route search."
    )

    st.divider()

    model_choice = st.radio(
        "Select Prediction Model",
        options=[
            "Compare All Three Models",
            "Linear Regression",
            "Random Forest",
            "HistGradientBoosting"
        ],
        horizontal=True
    )

    search_submitted = st.form_submit_button(
        "Find Trains and Predict Arrival Delay",
        use_container_width=True
    )


# ==================================================
# 7. FIND MATCHING TRAIN ROUTES
# ==================================================

if search_submitted:

    if source_station is None or destination_station is None:

        st.error("Please select both source and destination stations.")

        st.stop()

    if source_station == destination_station:

        st.error("Source and destination stations must be different.")

        st.stop()

    if category_choice == "PEXP":

        st.error(
            "Parcel Express is a non-passenger service. "
            "Please select a passenger train category."
        )

        st.stop()

    # Select the route matching keys
    run_columns = [
        "train_number",
        "train_name",
        "category",
        "start_date"
    ]

    # Source station records
    source_stops = passenger_routes.loc[
        passenger_routes["station_code"] == source_station,
        run_columns + ["seq"]
    ].copy()

    source_stops = source_stops.rename(
        columns={"seq": "source_sequence"}
    )

    # Destination station records
    destination_stops = passenger_routes.loc[
        (passenger_routes["station_code"] == destination_station)
        & passenger_routes["scheduled_arrival_hour"].notna(),
        run_columns
        + [
            "seq",
            "scheduled_arrival_hour",
            "scheduled_arrival_time"
        ]
    ].copy()

    destination_stops = destination_stops.rename(
        columns={
            "seq": "destination_sequence"
        }
    )

    # Match source and destination within the same train run
    matched_routes = source_stops.merge(
        destination_stops,
        on=run_columns,
        how="inner"
    )

    # Destination must occur after source
    matched_routes = matched_routes.loc[
        matched_routes["destination_sequence"]
        > matched_routes["source_sequence"]
    ].copy()

    # Apply category filter if selected
    if category_choice != "ALL":

        matched_routes = matched_routes.loc[
            matched_routes["category"] == category_choice
        ].copy()

    if matched_routes.empty:

        st.warning(
            f"No matching train services were found from "
            f"{source_station} to {destination_station} "
            f"in the available sample."
        )

        st.stop()


    # ==================================================
    # 8. SUMMARIZE HISTORICAL TRAIN SERVICES
    # ==================================================

    def most_common_value(series):

        values = series.dropna().mode()

        if len(values) == 0:
            return np.nan

        return values.iloc[0]


    route_summary = (
        matched_routes
        .groupby(
            [
                "train_number",
                "train_name",
                "category"
            ],
            as_index=False
        )
        .agg(
            destination_sequence=(
                "destination_sequence",
                most_common_value
            ),
            scheduled_arrival_hour=(
                "scheduled_arrival_hour",
                most_common_value
            ),
            scheduled_arrival_time=(
                "scheduled_arrival_time",
                most_common_value
            ),
            historical_runs=(
                "start_date",
                "nunique"
            ),
            first_sample_date=(
                "start_date",
                "min"
            ),
            last_sample_date=(
                "start_date",
                "max"
            )
        )
    )

    route_summary = route_summary.dropna(
        subset=[
            "destination_sequence",
            "scheduled_arrival_hour"
        ]
    ).copy()

    if route_summary.empty:

        st.warning(
            "Matching routes were found, but their destination "
            "arrival times were unavailable."
        )

        st.stop()

    route_summary["destination_sequence"] = (
        route_summary["destination_sequence"].astype(int)
    )

    route_summary["scheduled_arrival_hour"] = (
        route_summary["scheduled_arrival_hour"].astype(int)
    )


    # ==================================================
    # 9. PREPARE INPUT FEATURES FOR ALL ROUTES
    # ==================================================

    model_inputs = pd.DataFrame({

        "category": route_summary["category"].astype(str),

        "seq": route_summary["destination_sequence"],

        "station_code": destination_station,

        "scheduled_arrival_hour": (
            route_summary["scheduled_arrival_hour"]
        )
    })

    model_inputs = model_inputs[feature_columns]


    # ==================================================
    # 10. GENERATE PREDICTIONS
    # ==================================================

    predictions = {}

    # HistGradientBoosting uses its own Pipeline and
    # therefore receives the original DataFrame.
    if model_choice in [
        "Compare All Three Models",
        "HistGradientBoosting"
    ]:

        hist_predictions = models[
            "HistGradientBoosting"
        ].predict(model_inputs)

        predictions["HistGradientBoosting"] = (
            hist_predictions
        )

    # Linear Regression and Random Forest use
    # the saved One-Hot Encoder.
    if model_choice in [
        "Compare All Three Models",
        "Linear Regression",
        "Random Forest"
    ]:

        encoded_inputs = preprocessor.transform(
            model_inputs
        )

        selected_models = [
            "Linear Regression",
            "Random Forest"
        ]

        if model_choice != "Compare All Three Models":

            selected_models = [model_choice]

        for name in selected_models:

            predictions[name] = models[name].predict(
                encoded_inputs
            )


    # ==================================================
    # 11. DISPLAY ROUTE AND PREDICTION RESULTS
    # ==================================================

    st.divider()

    st.header("Available Train Services")

    st.write(
        f"Route: **{source_station} → {destination_station}**"
    )

    st.success(
        f"{len(route_summary)} matching train service(s) found."
    )

    st.caption(
        "These services are identified from the historical "
        "10% dataset sample. They are not confirmed live schedules."
    )

    # Build result table
    results = pd.DataFrame({

        "Train Number": route_summary["train_number"],

        "Train Name": route_summary["train_name"],

        "Category": route_summary["category"].apply(
            category_label
        ),

        "Destination Sequence": (
            route_summary["destination_sequence"]
        ),

        "Scheduled Arrival (IST)": (
            route_summary["scheduled_arrival_time"]
        ),

        "Historical Runs": (
            route_summary["historical_runs"]
        )
    })

    # Add selected model predictions
    for name, values in predictions.items():

        results[f"{name} Prediction (min)"] = np.round(
            values,
            2
        )

    # Sort by Linear Regression prediction when comparing all models
    if model_choice == "Compare All Three Models":

        results = results.sort_values(
            "Linear Regression Prediction (min)"
        )

    else:

        prediction_column = f"{model_choice} Prediction (min)"

        results = results.sort_values(
            prediction_column
        )

    results = results.reset_index(drop=True)

    # Display all matching train options and predictions
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )


    # ==================================================
    # 12. ROUTE SUMMARY
    # ==================================================

    st.divider()

    st.subheader("Route Search Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Matching Train Services",
        len(route_summary)
    )

    col2.metric(
        "Source Station",
        source_station
    )

    col3.metric(
        "Destination Station",
        destination_station
    )

    st.caption(
        "Historical Runs represents the number of distinct "
        "train start dates in the sample for each matching service."
    )

    st.info(
        "Predictions estimate the arrival delay at the selected "
        "destination station using train category, destination "
        "sequence, station code, and scheduled arrival hour. "
        "They are not live delay predictions."
    )


# ==================================================
# 13. FOOTER
# ==================================================

st.divider()

st.caption(
    "BookWisely | Machine Learning Based Train Arrival "
    "Delay Prediction | Educational Project"
)
