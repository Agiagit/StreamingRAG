import pandas as pd
import matplotlib.pyplot as plt


#  TESTDATEN 

data = [
    # --- Tests mit der "vorsichtigen" (cautious) Einstellung ---
    {"question": "Who won the men's 100m gold?", "setting": "cautious", "commit_text": "Who won the men's 100m", "confidence": 0.7265, "status": "correct"},
    {"question": "Where is the olympic village?", "setting": "cautious", "commit_text": "Where is the olympic village?", "confidence": 0.6558, "status": "too_late"},
    {"question": "How many medals did France win?", "setting": "cautious", "commit_text": "How many medals did France", "confidence": 0.813, "status": "correct"},
    {"question": "How many teams are competing in the basketball tournament?", "setting": "cautious", "commit_text": "How many teams are competing in the basketball tournament?", "confidence": 0.5865, "status": "too_late"},
    {"question": "Which athlete won the men's singles final in tennis?", "setting": "cautious", "commit_text": "Which athlete won the men's singles final in tennis?", "confidence": 0.719, "status": "too_late"},
    {"question": "Where are the 2024 Summer Olympics being held?", "setting": "cautious", "commit_text": "Where are the 2024 Summer Olympics being held?", "confidence": 0.553, "status": "too_late"},
    {"question": "Who won the gold medal in the newly added sport of breaking?", "setting": "cautious", "commit_text": "Who won the gold medal in the newly added sport of break", "confidence": 0.75, "status": "too_early"},
    {"question": "Which country did not qualify for the finals in men's tennis?", "setting": "cautious", "commit_text": "Which country did not qualify for the finals in men's tennis?", "confidence": 0.40, "status": "too_late"},
    {"question": "Where will the next Summer Olympics take place in 2028", "setting": "cautious", "commit_text": "Where will the next Summer Olympics take place in 2", "confidence": 0.734, "status": "too_early"},
    {"question": "How many gold medals did the host country win in the year 2020?", "setting": "cautious", "commit_text": "How many gold medals did the host country win in the year 2020?", "confidence": 0.566, "status": "too_late"},

    # --- Tests mit der "eifrigen" (eager) Einstellung ---
    {"question": "Who won the men's 100m gold?", "setting": "eager", "commit_text": "Who won", "confidence": 0.6, "status": "too_early"},
    {"question": "Where is the olympic village?", "setting": "eager", "commit_text": "Where is the olympic village", "confidence": 0.68, "status": "correct"},
    {"question": "How many medals did France win?", "setting": "eager", "commit_text": "How many medals did Fr", "confidence": 0.728, "status": "correct"},
    {"question": "How many teams are competing in the basketball tournament?", "setting": "eager", "commit_text": "How many teams are competing in the b", "confidence": 0.5828, "status": "too_early"},
    {"question": "Which athlete won the men's singles final in tennis?", "setting": "eager", "commit_text": "Which athlete won the men's singles final in te", "confidence": 0.7, "status": "too_early"},
    {"question": "Where are the 2024 Summer Olympics being held?", "setting": "eager", "commit_text": "Where are the 202", "confidence": 0.6044, "status": "too_early"},
    {"question": "Who won the gold medal in the newly added sport of breaking?", "setting": "eager", "commit_text": "Who won", "confidence": 0.6, "status": "too_early"},
    {"question": "Which country did not qualify for the finals in men's tennis?", "setting": "eager", "commit_text": "Which country did not qualify for the finals", "confidence": 0.58, "status": "too_early"},
    {"question": "Where will the next Summer Olympics take place in 2028", "setting": "eager", "commit_text": "Where will the next Summer Olym", "confidence": 0.595, "status": "too_early"},
    {"question": "How many gold medals did the host country win in the year 2020?", "setting": "eager", "commit_text": "How many gold", "confidence": 0.58, "status": "too_early"},
]


# AUSWERTUNG 

def run_analysis():
    print("Starte Evaluierung...\n")
    df = pd.DataFrame(data)
    
    # 1. Tabelle in der Konsole ausgeben
    print("--- Komplette Ergebnistabelle ---")
    print(df.to_string(index=False))
    print("\n")
    
    # 2. Zusammenfassung berechnen
    summary = df.groupby(['setting', 'status']).size().unstack(fill_value=0)
    
    # Sicherstellen, dass alle Spalten existieren, auch wenn sie 0 sind
    for col in ['too_early', 'correct', 'too_late']:
        if col not in summary.columns:
            summary[col] = 0
            
    # Spalten sortieren für schönere Darstellung
    summary = summary[['too_early', 'correct', 'too_late']]
    
    print("--- Zusammenfassung (Anzahl der Entscheidungen) ---")
    print(summary)
    print("\n")
    
    # 3. DIAGRAMME GENERIEREN
   
    # Balkendiagramm erstellen
    ax = summary.plot(
        kind='bar', 
        stacked=False, 
        figsize=(8, 5), 
        color=['#ff9999', '#66b3ff', '#99ff99']
    )
    
    plt.title('Systemverhalten: Cautious vs. Eager Thresholds')
    plt.xlabel('Einstellung (Setting)')
    plt.ylabel('Anzahl der Fragen')
    plt.xticks(rotation=0)
    plt.legend(title='Entscheidungs-Status', labels=['Zu früh', 'Korrekt', 'Zu spät'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Diagramm als Bild speichern und anzeigen
    plt.tight_layout()
    plt.savefig('evaluation_chart.png')
    print("Diagramm wurde als 'evaluation_chart.png' gespeichert.")
    
    plt.show()

if __name__ == "__main__":
    run_analysis()