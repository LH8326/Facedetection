import cv2
import pathlib

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

print(f"Using Haar cascade from: {cascade_path}")

# Load the Haar cascade
clf = cv2.CascadeClassifier(str(cascade_path))

# Start video capture
# camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture("video.mp4")

while True:  # Fixed 'true' to 'True'
    ret, frame = camera.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around faces
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)  # Fixed missing comma and closing parenthesis

    # Show the result
    cv2.imshow("Faces", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()


