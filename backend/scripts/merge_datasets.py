# scripts/merge_datasets.py

import os
import sys

import pandas as pd

# 1) Sanity-check that all source files exist and are non-empty
for fn in ("data/mental_health_survey.csv", "data/phq9.csv", "data/sleep_health.csv"):
    if not os.path.exists(fn):
        print(f"❌ Missing file: {fn}")
        sys.exit(1)
    if os.path.getsize(fn) == 0:
        print(f"❌ Empty file: {fn}")
        sys.exit(1)


def load_and_clean_survey(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # keep only adults
    df = df[df["Age"].between(18, 100)]
    # rename core columns
    df = df.rename(
        columns={
            "Age": "age",
            "Gender": "gender",
            "treatment": "treatment_raw",
            "work_interfere": "work_interfere_raw",
        }
    )

    # normalize gender text
    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.lower()
        .map(lambda x: "male" if "male" in x else ("female" if "female" in x else "other"))
    )

    # map binary survey columns: Yes=1, No=0, Don't know / Not sure / Maybe = 2
    binary_cols = [
        "self_employed",
        "family_history",
        "benefits",
        "care_options",
        "wellness_program",
        "seek_help",
        "anonymity",
        "leave",
        "mental_health_consequence",
        "phys_health_consequence",
        "coworkers",
        "supervisor",
        "mental_health_interview",
        "phys_health_interview",
        "mental_vs_physical",
        "obs_consequence",
    ]
    df[binary_cols] = df[binary_cols].replace(
        {
            "Yes": 1,
            "No": 0,
            "Don't know": 2,
            "Don’t know": 2,
            "Not sure": 2,
            "Maybe": 2,
        }
    )

    # map work_interfere to ordinal: Never=0, Rarely=1, Sometimes=2, Often=3
    wi_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
    df["work_interfere"] = (
        df["work_interfere_raw"].map(wi_map).fillna(0).astype(int)
    )

    # keep only the columns we need from the survey
    keep_cols = [
        "age",
        "gender",
        *binary_cols,
        "work_interfere",
        "treatment_raw",
    ]
    return df[keep_cols]


def load_and_clean_phq(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure age & gender columns exist
    df = df.rename(columns={"age": "age", "gender": "gender"})
    df = df[df["age"].between(18, 100)]
    # normalize gender
    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.lower()
        .map(lambda x: "male" if "male" in x else ("female" if "female" in x else "other"))
    )
    # assume phq9_score exists and numeric
    df["phq9_score"] = pd.to_numeric(df["phq9_score"], errors="coerce")
    df = df.dropna(subset=["phq9_score", "age", "gender"])
    return df[["age", "gender", "phq9_score"]]


def load_and_clean_sleep(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure age & gender columns exist
    df = df.rename(columns={"age": "age", "gender": "gender"})
    df = df[df["age"].between(18, 100)]
    # normalize gender
    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.lower()
        .map(lambda x: "male" if "male" in x else ("female" if "female" in x else "other"))
    )
    # assume sleep_hours exists and numeric
    df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")
    df = df.dropna(subset=["sleep_hours", "age", "gender"])
    return df[["age", "gender", "sleep_hours"]]


def assign_bins(df: pd.DataFrame) -> pd.DataFrame:
    # assign age_group bins 18–25, 26–35, 36–50, 51–65, 66–100
    df["age_group"] = pd.cut(
        df["age"], bins=[17, 25, 35, 50, 65, 100], labels=False
    )
    return df


def main():
    # 1) load & clean
    survey = load_and_clean_survey("data/mental_health_survey.csv")
    phq = load_and_clean_phq("data/phq9.csv")
    sleep = load_and_clean_sleep("data/sleep_health.csv")

    # 2) assign age_group and gender_norm
    for df in (survey, phq, sleep):
        df[:] = assign_bins(df)

    # 3) merge in phq9 mean by bin
    phq_grouped = (
        phq.groupby(["age_group", "gender"])[["phq9_score"]]
        .mean()
        .reset_index()
    )
    survey = survey.merge(
        phq_grouped, on=["age_group", "gender"], how="left"
    )

    # 4) merge in sleep_hours mean by bin
    sleep_grouped = (
        sleep.groupby(["age_group", "gender"])[["sleep_hours"]]
        .mean()
        .reset_index()
    )
    survey = survey.merge(
        sleep_grouped, on=["age_group", "gender"], how="left"
    )

    # 5) create final label (treatment 0/1/2)
    treatment_map = {"Yes": 1, "No": 0, "Don't know": 2, "Don’t know": 2}
    survey["label"] = survey["treatment_raw"].map(treatment_map).fillna(2).astype(int)

    # 6) drop helper columns
    drop_cols = ["age_group", "treatment_raw"]
    survey = survey.drop(columns=drop_cols)

    # 7) save
    os.makedirs("data", exist_ok=True)
    survey.to_csv("data/dataset.csv", index=False)
    print(f"✅ Merged dataset shape: {survey.shape} → data/dataset.csv")


if __name__ == "__main__":
    main()
# scripts/merge_datasets.py

import os
import sys

import pandas as pd

# 1) Sanity-check that all source files exist and are non-empty
for fn in ("data/mental_health_survey.csv", "data/phq9.csv", "data/sleep_health.csv"):
    if not os.path.exists(fn):
        print(f"❌ Missing file: {fn}")
        sys.exit(1)
    if os.path.getsize(fn) == 0:
        print(f"❌ Empty file: {fn}")
        sys.exit(1)


def load_and_clean_survey(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # keep only adults
    df = df[df["Age"].between(18, 100)]
    # rename core columns
    df = df.rename(
        columns={
            "Age": "age",
            "Gender": "gender",
            "treatment": "treatment_raw",
            "work_interfere": "work_interfere_raw",
        }
    )

    # normalize gender text
    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.lower()
        .map(lambda x: "male" if "male" in x else ("female" if "female" in x else "other"))
    )

    # map binary survey columns: Yes=1, No=0, Don't know / Not sure / Maybe = 2
    binary_cols = [
        "self_employed",
        "family_history",
        "benefits",
        "care_options",
        "wellness_program",
        "seek_help",
        "anonymity",
        "leave",
        "mental_health_consequence",
        "phys_health_consequence",
        "coworkers",
        "supervisor",
        "mental_health_interview",
        "phys_health_interview",
        "mental_vs_physical",
        "obs_consequence",
    ]
    df[binary_cols] = df[binary_cols].replace(
        {
            "Yes": 1,
            "No": 0,
            "Don't know": 2,
            "Don’t know": 2,
            "Not sure": 2,
            "Maybe": 2,
        }
    )

    # map work_interfere to ordinal: Never=0, Rarely=1, Sometimes=2, Often=3
    wi_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
    df["work_interfere"] = (
        df["work_interfere_raw"].map(wi_map).fillna(0).astype(int)
    )

    # keep only the columns we need from the survey
    keep_cols = [
        "age",
        "gender",
        *binary_cols,
        "work_interfere",
        "treatment_raw",
    ]
    return df[keep_cols]


def load_and_clean_phq(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure age & gender columns exist
    df = df.rename(columns={"age": "age", "gender": "gender"})
    df = df[df["age"].between(18, 100)]
    # normalize gender
    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.lower()
        .map(lambda x: "male" if "male" in x else ("female" if "female" in x else "other"))
    )
    # assume phq9_score exists and numeric
    df["phq9_score"] = pd.to_numeric(df["phq9_score"], errors="coerce")
    df = df.dropna(subset=["phq9_score", "age", "gender"])
    return df[["age", "gender", "phq9_score"]]


def load_and_clean_sleep(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure age & gender columns exist
    df = df.rename(columns={"age": "age", "gender": "gender"})
    df = df[df["age"].between(18, 100)]
    # normalize gender
    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.lower()
        .map(lambda x: "male" if "male" in x else ("female" if "female" in x else "other"))
    )
    # assume sleep_hours exists and numeric
    df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")
    df = df.dropna(subset=["sleep_hours", "age", "gender"])
    return df[["age", "gender", "sleep_hours"]]


def assign_bins(df: pd.DataFrame) -> pd.DataFrame:
    # assign age_group bins 18–25, 26–35, 36–50, 51–65, 66–100
    df["age_group"] = pd.cut(
        df["age"], bins=[17, 25, 35, 50, 65, 100], labels=False
    )
    return df


def main():
    # 1) load & clean
    survey = load_and_clean_survey("data/mental_health_survey.csv")
    phq = load_and_clean_phq("data/phq9.csv")
    sleep = load_and_clean_sleep("data/sleep_health.csv")

    # 2) assign age_group and gender_norm
    for df in (survey, phq, sleep):
        df[:] = assign_bins(df)

    # 3) merge in phq9 mean by bin
    phq_grouped = (
        phq.groupby(["age_group", "gender"])[["phq9_score"]]
        .mean()
        .reset_index()
    )
    survey = survey.merge(
        phq_grouped, on=["age_group", "gender"], how="left"
    )

    # 4) merge in sleep_hours mean by bin
    sleep_grouped = (
        sleep.groupby(["age_group", "gender"])[["sleep_hours"]]
        .mean()
        .reset_index()
    )
    survey = survey.merge(
        sleep_grouped, on=["age_group", "gender"], how="left"
    )

    # 5) create final label (treatment 0/1/2)
    treatment_map = {"Yes": 1, "No": 0, "Don't know": 2, "Don’t know": 2}
    survey["label"] = survey["treatment_raw"].map(treatment_map).fillna(2).astype(int)

    # 6) drop helper columns
    drop_cols = ["age_group", "treatment_raw"]
    survey = survey.drop(columns=drop_cols)

    # 7) save
    os.makedirs("data", exist_ok=True)
    survey.to_csv("data/dataset.csv", index=False)
    print(f"✅ Merged dataset shape: {survey.shape} → data/dataset.csv")


if __name__ == "__main__":
    main()
# scripts/merge_datasets.py

import os
import sys

import pandas as pd

# 1) Sanity-check that all source files exist and are non-empty
for fn in ("data/mental_health_survey.csv", "data/phq9.csv", "data/sleep_health.csv"):
    if not os.path.exists(fn):
        print(f"❌ Missing file: {fn}")
        sys.exit(1)
    if os.path.getsize(fn) == 0:
        print(f"❌ Empty file: {fn}")
        sys.exit(1)


def load_and_clean_survey(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # keep only adults
    df = df[df["Age"].between(18, 100)]
    # rename core columns
    df = df.rename(
        columns={
            "Age": "age",
            "Gender": "gender",
            "treatment": "treatment_raw",
            "work_interfere": "work_interfere_raw",
        }
    )

    # normalize gender text
    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.lower()
        .map(lambda x: "male" if "male" in x else ("female" if "female" in x else "other"))
    )

    # map binary survey columns: Yes=1, No=0, Don't know / Not sure / Maybe = 2
    binary_cols = [
        "self_employed",
        "family_history",
        "benefits",
        "care_options",
        "wellness_program",
        "seek_help",
        "anonymity",
        "leave",
        "mental_health_consequence",
        "phys_health_consequence",
        "coworkers",
        "supervisor",
        "mental_health_interview",
        "phys_health_interview",
        "mental_vs_physical",
        "obs_consequence",
    ]
    df[binary_cols] = df[binary_cols].replace(
        {
            "Yes": 1,
            "No": 0,
            "Don't know": 2,
            "Don’t know": 2,
            "Not sure": 2,
            "Maybe": 2,
        }
    )

    # map work_interfere to ordinal: Never=0, Rarely=1, Sometimes=2, Often=3
    wi_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
    df["work_interfere"] = (
        df["work_interfere_raw"].map(wi_map).fillna(0).astype(int)
    )

    # keep only the columns we need from the survey
    keep_cols = [
        "age",
        "gender",
        *binary_cols,
        "work_interfere",
        "treatment_raw",
    ]
    return df[keep_cols]


def load_and_clean_phq(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure age & gender columns exist
    df = df.rename(columns={"age": "age", "gender": "gender"})
    df = df[df["age"].between(18, 100)]
    # normalize gender
    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.lower()
        .map(lambda x: "male" if "male" in x else ("female" if "female" in x else "other"))
    )
    # assume phq9_score exists and numeric
    df["phq9_score"] = pd.to_numeric(df["phq9_score"], errors="coerce")
    df = df.dropna(subset=["phq9_score", "age", "gender"])
    return df[["age", "gender", "phq9_score"]]


def load_and_clean_sleep(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure age & gender columns exist
    df = df.rename(columns={"age": "age", "gender": "gender"})
    df = df[df["age"].between(18, 100)]
    # normalize gender
    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.lower()
        .map(lambda x: "male" if "male" in x else ("female" if "female" in x else "other"))
    )
    # assume sleep_hours exists and numeric
    df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")
    df = df.dropna(subset=["sleep_hours", "age", "gender"])
    return df[["age", "gender", "sleep_hours"]]


def assign_bins(df: pd.DataFrame) -> pd.DataFrame:
    # assign age_group bins 18–25, 26–35, 36–50, 51–65, 66–100
    df["age_group"] = pd.cut(
        df["age"], bins=[17, 25, 35, 50, 65, 100], labels=False
    )
    return df


def main():
    # 1) load & clean
    survey = load_and_clean_survey("data/mental_health_survey.csv")
    phq = load_and_clean_phq("data/phq9.csv")
    sleep = load_and_clean_sleep("data/sleep_health.csv")

    # 2) assign age_group and gender_norm
    for df in (survey, phq, sleep):
        df[:] = assign_bins(df)

    # 3) merge in phq9 mean by bin
    phq_grouped = (
        phq.groupby(["age_group", "gender"])[["phq9_score"]]
        .mean()
        .reset_index()
    )
    survey = survey.merge(
        phq_grouped, on=["age_group", "gender"], how="left"
    )

    # 4) merge in sleep_hours mean by bin
    sleep_grouped = (
        sleep.groupby(["age_group", "gender"])[["sleep_hours"]]
        .mean()
        .reset_index()
    )
    survey = survey.merge(
        sleep_grouped, on=["age_group", "gender"], how="left"
    )

    # 5) create final label (treatment 0/1/2)
    treatment_map = {"Yes": 1, "No": 0, "Don't know": 2, "Don’t know": 2}
    survey["label"] = survey["treatment_raw"].map(treatment_map).fillna(2).astype(int)

    # 6) drop helper columns
    drop_cols = ["age_group", "treatment_raw"]
    survey = survey.drop(columns=drop_cols)

    # 7) save
    os.makedirs("data", exist_ok=True)
    survey.to_csv("data/dataset.csv", index=False)
    print(f"✅ Merged dataset shape: {survey.shape} → data/dataset.csv")


if __name__ == "__main__":
    main()
