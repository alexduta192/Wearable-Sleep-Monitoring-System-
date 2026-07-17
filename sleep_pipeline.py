"""
questionnaire_pipeline.py
=========================
1. Creeaza tabelul 'questionnaire' in SQLite
2. Importa raspunsurile din CSV exportat din Google Forms
3. Coreleaza raspunsurile subiective cu datele obiective din senzori
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DB_PATH    = "/home/pi/raw_sensor_data.db"
OUTPUT_DIR = "sleep_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS questionnaire (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT,
    session_id              TEXT,
    subject_name            TEXT,
    age                     INTEGER,
    sex                     TEXT,
    bedtime                 TEXT,
    waketime                TEXT,
    sleep_quality           INTEGER,
    wakeups_count           TEXT,
    wakeup_reason           TEXT,
    sleep_onset             TEXT,
    stress_level            INTEGER,
    movement_felt           TEXT,
    refreshed_feeling       INTEGER,
    temperature_comfort     TEXT,
    had_dreams              TEXT,
    snored                  TEXT,
    used_alarm              TEXT,
    consumed_caffeine       INTEGER DEFAULT 0,
    consumed_heavy_meal     INTEGER DEFAULT 0,
    consumed_alcohol        INTEGER DEFAULT 0,
    consumed_supplements    INTEGER DEFAULT 0,
    consumed_screen         INTEGER DEFAULT 0,
    consumed_exercise       INTEGER DEFAULT 0,
    additional_comments     TEXT
);
"""

COLUMN_MAP = {
    "Timestamp":                                                      "timestamp",
    "Session ID (provided by the researcher)":                        "session_id",
    "Name:":                                                          "subject_name",
    "Age:":                                                           "age",
    "Sex:":                                                           "sex",
    "What time did you go to bed?":                                   "bedtime",
    "What time did you wake up?":                                     "waketime",
    "How would you rate your overall sleep quality?":                 "sleep_quality",
    "How many times did you wake up during the night?":               "wakeups_count",
    "If you woke up during the night, what was the reason?":          "wakeup_reason",
    "How long did it take you to fall asleep?":                       "sleep_onset",
    "What was your stress level before going to sleep?":              "stress_level",
    "Did you feel that you moved a lot during sleep?":                "movement_felt",
    "How refreshed do you feel after waking up?":                     "refreshed_feeling",
    "Did you feel too hot or too cold during the night?":             "temperature_comfort",
    "Do you remember having dreams during the night?":                "had_dreams",
    "Did you snore during the night (or were you told you snored)?":  "snored",
    "Did you used an alarm clock to wake up?":                        "used_alarm",
    "What did you consume or do before going to sleep?":              "consumed_raw",
    "Any additional comments about your sleep?":                      "additional_comments",
}


def create_table(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()
    conn.close()
    print("[DB] Tabel 'questionnaire' creat/verificat.")


def import_from_csv(csv_path, db_path):
    if not os.path.exists(csv_path):
        print(f"EROARE: '{csv_path}' nu a fost gasit!")
        return 0

    df = pd.read_csv(csv_path)
    print(f"[CSV] {len(df)} raspunsuri gasite")
    conn = sqlite3.connect(db_path)
    imported = 0

    for _, row in df.iterrows():
        record = {}
        for csv_col, db_col in COLUMN_MAP.items():
            if csv_col in df.columns:
                val = row.get(csv_col, None)
                record[db_col] = None if pd.isna(val) else val

        consumed_raw = record.pop("consumed_raw", "") or ""
        record["consumed_caffeine"]    = 1 if "Caffeine"   in str(consumed_raw) else 0
        record["consumed_heavy_meal"]  = 1 if "Heavy meal" in str(consumed_raw) else 0
        record["consumed_alcohol"]     = 1 if "Alcohol"    in str(consumed_raw) else 0
        record["consumed_supplements"] = 1 if "supplement" in str(consumed_raw).lower() else 0
        record["consumed_screen"]      = 1 if "Screen"     in str(consumed_raw) else 0
        record["consumed_exercise"]    = 1 if "exercise"   in str(consumed_raw).lower() else 0

        session_id = record.get("session_id", "")
        existing = conn.execute("SELECT id FROM questionnaire WHERE session_id=?", (session_id,)).fetchone()
        if existing:
            print(f"  [SKIP] '{session_id}' deja importat")
            continue

        cols = ", ".join(record.keys())
        placeholders = ", ".join(["?"] * len(record))
        try:
            conn.execute(f"INSERT INTO questionnaire ({cols}) VALUES ({placeholders})", list(record.values()))
            imported += 1
            print(f"  [OK] {session_id} – {record.get('subject_name','N/A')}")
        except Exception as e:
            print(f"  [ERR] {session_id}: {e}")

    conn.commit()
    conn.close()
    print(f"[CSV] {imported} raspunsuri importate")
    return imported


def load_objective_scores():
    scores = []
    if not os.path.exists(OUTPUT_DIR):
        return pd.DataFrame()
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("score_") and f.endswith(".json"):
            session_id = f.replace("score_","").replace(".json","")
            with open(os.path.join(OUTPUT_DIR, f)) as fp:
                data = json.load(fp)
            data["session_id"] = session_id
            scores.append(data)
    if not scores:
        return pd.DataFrame()
    rows = []
    for s in scores:
        row = {"session_id": s["session_id"],
               "objective_score": s.get("total_score", np.nan),
               "duration_min": s.get("session_duration_min", np.nan)}
        for k, v in s.get("components", {}).items():
            row[f"obj_{k}"] = v
        for k, v in s.get("stage_distribution_pct", {}).items():
            row[f"pct_{k.replace(' ','_').replace('/','_')}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def correlate(db_path):
    conn = sqlite3.connect(db_path)
    try:
        q_df = pd.read_sql_query("SELECT * FROM questionnaire", conn)
    except:
        q_df = pd.DataFrame()
    conn.close()

    o_df = load_objective_scores()

    if q_df.empty:
        print("EROARE: Nu exista raspunsuri la chestionar in DB!")
        return None
    if o_df.empty:
        print("EROARE: Nu exista scoruri obiective! Ruleaza mai intai sleep_pipeline.py")
        return None

    merged = pd.merge(q_df, o_df, on="session_id", how="inner")
    print(f"[CORR] {len(merged)} sesiuni cu ambele tipuri de date")

    if merged.empty:
        print(f"  Sesiuni chestionar: {q_df['session_id'].tolist()}")
        print(f"  Sesiuni obiective:  {o_df['session_id'].tolist()}")
        return None

    merged["subjective_quality"]   = pd.to_numeric(merged["sleep_quality"], errors="coerce")
    merged["subjective_refreshed"] = pd.to_numeric(merged["refreshed_feeling"], errors="coerce")
    merged["subjective_stress"]    = pd.to_numeric(merged["stress_level"], errors="coerce")
    merged["wakeups_numeric"]      = merged["wakeups_count"].map({"0":0,"1":1,"2":2,"3":3,"More than 3":4})
    merged["movement_numeric"]     = merged["movement_felt"].map({"Not at all":0,"A little":1,"Moderately":2,"A lot":3})
    merged["snore_numeric"]        = merged["snored"].map({"No":0,"Yes, mildly":1,"Yes, frequently":2,"I do not know":0})

    merged["subjective_score"] = (
        merged["subjective_quality"].fillna(3) * 20 +
        merged["subjective_refreshed"].fillna(3) * 10 -
        merged["subjective_stress"].fillna(3) * 5 -
        merged["wakeups_numeric"].fillna(1) * 5
    ).clip(0, 100)

    print("\n[CORR] Scor subiectiv vs obiectiv per sesiune:")
    for _, row in merged.iterrows():
        print(f"  {row['session_id'][:30]:30} | Subiectiv: {row['subjective_score']:.0f}/100 | Obiectiv: {row['objective_score']:.1f}/100")

             
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0D1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    plots = [
        ("subjective_quality",   "objective_score",  "Calitate subiectiva vs Scor obiectiv"),
        ("subjective_refreshed", "objective_score",  "Senzatie odihnit vs Scor obiectiv"),
        ("subjective_stress",    "obj_ppg_quality",  "Stres pre-somn vs Calitate PPG"),
        ("wakeups_numeric",      "obj_architecture", "Treziri vs Arhitectura somn"),
        ("movement_numeric",     "obj_movement",     "Miscare perceputa vs Obiectiva"),
        ("snore_numeric",        "obj_snoring",      "Sforait raportat vs Obiectiv"),
    ]

    for idx, (x_col, y_col, title) in enumerate(plots):
        ax = fig.add_subplot(gs[idx//3, idx%3])
        ax.set_facecolor("#0D1117")
        if x_col in merged.columns and y_col in merged.columns:
            x = merged[x_col].dropna()
            y = merged[y_col].dropna()
            common = x.index.intersection(y.index)
            x, y = x[common], y[common]
            if len(x) >= 2:
                ax.scatter(x, y, color="#4FC3F7", s=80, alpha=0.8, edgecolors="#1565C0")
                z = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(x_line, np.poly1d(z)(x_line), color="#EF5350", linewidth=1.5, linestyle="--")
                r = x.corr(y)
                ax.text(0.05, 0.92, f"r = {r:+.3f}", transform=ax.transAxes, color="#FFD54F", fontsize=9, fontweight="bold")
            else:
                ax.text(0.5, 0.5, f"Prea putine date\n({len(x)} sesiuni)", ha="center", va="center", color="gray", transform=ax.transAxes)
        ax.set_title(title, color="white", fontsize=8)
        ax.tick_params(colors="gray", labelsize=7)
        ax.spines[:].set_color("#333333")
        ax.set_xlabel(x_col.replace("_"," "), color="gray", fontsize=7)
        ax.set_ylabel(y_col.replace("_"," "), color="gray", fontsize=7)

    plt.suptitle("Corelatie Subiectiv (Chestionar) vs Obiectiv (Senzori)", color="white", fontsize=12, fontweight="bold")
    out = os.path.join(OUTPUT_DIR, "correlation_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[PLOT] {out}")

                 
    rpt = os.path.join(OUTPUT_DIR, "correlation_report.txt")
    with open(rpt, "w") as f:
        f.write("="*60+"\n  RAPORT CORELATIE SUBIECTIV vs OBIECTIV\n"+"="*60+"\n\n")
        for _, row in merged.iterrows():
            f.write(f"Sesiune: {row['session_id']}\n")
            f.write(f"Subiect: {row.get('subject_name','N/A')} | Varsta: {row.get('age','N/A')} | Sex: {row.get('sex','N/A')}\n")
            f.write(f"  Calitate somn:     {row.get('sleep_quality','N/A')}/5\n")
            f.write(f"  Senzatie odihnit:  {row.get('refreshed_feeling','N/A')}/5\n")
            f.write(f"  Stres pre-somn:    {row.get('stress_level','N/A')}/5\n")
            f.write(f"  Treziri:           {row.get('wakeups_count','N/A')}\n")
            f.write(f"  Sforait:           {row.get('snored','N/A')}\n")
            consumed = [k.replace("consumed_","") for k in ["consumed_caffeine","consumed_heavy_meal","consumed_alcohol","consumed_supplements","consumed_screen","consumed_exercise"] if row.get(k)]
            f.write(f"  Consum pre-somn:   {', '.join(consumed) or 'Nimic'}\n")
            f.write(f"  Scor obiectiv:     {row.get('objective_score','N/A'):.1f}/100\n")
            f.write(f"  Scor subiectiv:    {row.get('subjective_score','N/A'):.0f}/100\n")
            diff = abs(row.get('subjective_score',0) - row.get('objective_score',0))
            f.write(f"  Acord S-O:         {'Bun' if diff<15 else 'Moderat' if diff<30 else 'Discrepanta mare'} (diff={diff:.0f}pt)\n\n")
    print(f"[REPORT] {rpt}")
    print(f"\n✅  Analiza completa! Fisierele sunt in: ./{OUTPUT_DIR}/")
    return merged


def print_usage():
    print("""
Utilizare:
  python3 questionnaire_pipeline.py setup
  python3 questionnaire_pipeline.py import responses.csv
  python3 questionnaire_pipeline.py correlate
  python3 questionnaire_pipeline.py all responses.csv

Cum exporti CSV din Google Forms:
  Forms -> Responses -> iconica Sheets -> File -> Download -> CSV
  Apoi: scp responses.csv pi@YOUR_SERVER_IP:/home/pi/
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)
    cmd = sys.argv[1].lower()
    if cmd == "setup":
        create_table(DB_PATH)
    elif cmd == "import":
        if len(sys.argv) < 3:
            print("EROARE: python3 questionnaire_pipeline.py import responses.csv")
            sys.exit(1)
        create_table(DB_PATH)
        import_from_csv(sys.argv[2], DB_PATH)
    elif cmd == "correlate":
        correlate(DB_PATH)
    elif cmd == "all":
        if len(sys.argv) < 3:
            print("EROARE: python3 questionnaire_pipeline.py all responses.csv")
            sys.exit(1)
        create_table(DB_PATH)
        import_from_csv(sys.argv[2], DB_PATH)
        correlate(DB_PATH)
    else:
        print_usage()
