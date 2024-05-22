QUIT = "QUIT"
YES = "YES"
NO = "NO"
SIZE_RECEIVED = "SIZE_RECEIVED"
NO_CLASS_IN_SESSION = "No class is currently ongoing in this room. Please try again later."

SUCCESSFULLY_MARKED_PRESENT = lambda student_id, subject,section : f"Student {student_id} was successfully marked present for {subject}, section {section}"
NOT_ENROLLED = lambda student_id,subject,section  : f"Student {student_id} is not enrolled in {subject}, section {section}"

ALREADY_SEEN = lambda student_id : f"Already checked student: {student_id}"

if __name__ == "__main__":
    print(NOT_ENROLLED(20180596,'Calculus 1',4))
    print(SUCCESSFULLY_MARKED_PRESENT(19820382,'Calculus 1',4))
    print(NO_CLASS_IN_SESSION)