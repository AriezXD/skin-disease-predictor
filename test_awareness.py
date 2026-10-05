from src.ai_awareness import get_disease_awareness


disease = "psoriasis"
confidence = 0.8835


print("Sending prediction to Gemini...")
print(f"Disease: {disease}")
print(f"Model score: {confidence * 100:.2f}%")
print()


result = get_disease_awareness(
    disease=disease,
    confidence=confidence
)


print("===== AI AWARENESS =====")

print("\nDisease:")
print(disease)

print("\nWhat is it?")
print(result.what_is_it)

print("\nPossible contributing factors:")
for factor in result.possible_factors:
    print(f"- {factor}")

print("\nWhat to do next:")
for step in result.what_to_do_next:
    print(f"- {step}")

print("\nWhen to seek professional help:")
print(f"Urgency: {result.seek_help.urgency}")
print(f"Recommended: {result.seek_help.recommended}")

for reason in result.seek_help.reasons:
    print(f"- {reason}")

print("\nImportant note:")
print(result.important_note)