YOLO SESSION
AI Project Briefs
Five real-world Computer Vision projects using pretrained YOLO models

Choose ONE project from the list below.
Read the full brief for your project before writing a single line of code.

P1  Near-Miss Traffic Analytics

Smart City Road Safety System

P2  Classroom Engagement

Analytics

EdTech Attention & Behaviour System

P3  Elderly Fall Detection

Healthcare AI Emergency Response System

P4  Retail Customer
Intelligence

Business Analytics & Store Insights

P5  Crowd Crush Prevention

Public Safety & Stampede Prediction

Rules for All Projects

▸  Use only the provided pretrained YOLO models — no training, no fine-tuning, no other pretrained models.
▸  CPU users: use yolo26n models. GPU users: you may use yolo26m for better accuracy.
▸  LLMs (ChatGPT, Claude, etc.) may only be used for debugging, syntax help, and reading documentation.
▸  You may NOT use an LLM to design your solution logic, risk formula, or scoring system.
▸  Every Designer's Choice section must be answered in your submission. Your reasoning is graded.
▸  Your system must produce an annotated output video that can be played back.

Models provided: yolo26n.pt  ·  yolo26m.pt  ·  yolo26n-pose.pt  ·  yolo26n-seg.pt  ·  yolo26n-obb.pt  ·  yolo26n-cls.pt

PROJECT  1
Near-Miss Traffic Analytics
Smart City Road Safety System

Overview:  You are an AI engineer for a Smart City Department. Your task is to build a system that watches traffic
surveillance video and automatically identifies near-miss events — moments where a collision between road users
nearly occurred. The goal is not just detection: it is intelligent risk analysis.

Model to Use
Primary (CPU): yolo26n.pt
Better accuracy (GPU): yolo26m.pt
Detects: person, bicycle, car,
motorcycle, bus, truck — all COCO classes.

Video Source
Pexels — search: traffic intersection
https://www.pexels.com/search/videos/traffic%20interse
ction/
Pixabay — search: traffic intersection
UA-DETRAC (Kaggle): real CCTV, 100 videos
YouTube: https://youtu.be/KBsqQez-O4w
TIP: Use overhead or elevated angle footage.

STAGE  1

Object Detection — Find Every Road User

Load your model and run detection on every frame of the traffic video. You should only detect road users — not
background objects.

💡  Implementation

▸  model = YOLO('yolo26n.pt')
▸  results = model(frame, conf=0.4, classes=[0,1,2,3,5,7])
▸  classes: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
▸  Draw bounding box, class name, and confidence score on each detection.

STAGE  2

Object Tracking — Give Each Object an Identity

Detection alone is not enough. You need to track each object so that Car #5 in frame 10 is still Car #5 in frame 100.
This is essential for measuring movement and proximity over time.

💡  Implementation

▸  results = model.track(frame, persist=True, tracker='bytetrack.yaml')
▸  Access track IDs: r.boxes.id (may be None if no tracks yet)
▸  Store history: for each track_id, keep a list of the last 30 center positions.
▸  Display track ID on the bounding box alongside class name.

STAGE  3

Proximity Detection — Which Objects Are Interacting?

Every frame, check all pairs of tracked objects where at least one is a vehicle. Determine whether they are close
enough to be considered an interacting pair.

🎨  DESIGNER'S CHOICE — How Do You Measure 'Too Close'?
There is no single correct answer. You must choose a method and justify it.

Consider:

▸  Option A — Euclidean distance between bounding box centers (simplest, ignores object size)
▸  Option B — Overlap of expanded bounding boxes (scale boxes by 1.5x, check IOU > 0)
▸  Option C — Distance between nearest edges of bounding boxes (more realistic than centers)
▸  Problem to consider: objects far from the camera appear smaller — a fixed pixel threshold will miss
danger at distance. How will you account for this?
▸  Should person-person pairs count? Or only vehicle-involving pairs?

📝  Document your reasoning — this carries marks.

STAGE  4

Risk Scoring — How Dangerous Is This Interaction?

For each interacting pair, calculate a risk score. This is the intelligence layer of your system. The score should reflect
how dangerous the situation is — not just how close two objects are.

🎨  DESIGNER'S CHOICE — Design Your Risk Formula
Risk is not just about distance. Think about what actually makes a situation dangerous on a road.

Consider:

▸  Factor: Distance — closer = higher risk. But how does it scale? Linearly? Exponentially?
▸  Factor: Object type combination — car + pedestrian vs car + car. Do these have equal risk?
▸  Factor: Relative velocity — are they moving toward each other? Use the track history (position
change per frame) to estimate speed and direction.
▸  Factor: Time to potential impact — can you estimate how many frames until they would collide at
current speed?
▸  Output format: three levels (Low / Medium / High) OR a numerical score 0–100.
▸  Define your exact threshold values and explain why you chose them.

📝  Document your reasoning — this carries marks.

STAGE  5

Near-Miss Detection — Define the Event

A near-miss is not just one frame of high risk. It is a sustained dangerous interaction. Your system must decide when a
sequence of high-risk frames constitutes an actual near-miss event.

🎨  DESIGNER'S CHOICE — What Is a Near-Miss?

This definition is yours to make. Two different engineers may define it differently — both can be correct if
well-reasoned.

Consider:

▸  Duration: must the risk stay high for N consecutive frames? What value of N is meaningful?
▸  Recovery: if risk drops briefly then spikes again, is that one event or two?
▸  Direction: should both objects be moving toward each other, or is proximity alone enough?
▸  Consider: a parked car near a pedestrian would always be 'close' — how do you filter this out?
▸  Once a near-miss is logged, define a cooldown period before the same pair can trigger another
event.

📝  Document your reasoning — this carries marks.

STAGE  6

Alert Generation & Dashboard

Visually communicate the system's analysis. Normal detections, interactions, and near-miss events should all look
different at a glance.

💡  Visual Alert Design

▸  Normal objects: green bounding box
▸  Interacting pair: yellow bounding box
▸  Near-miss event: red bounding box + flashing '⚠ NEAR-MISS' text overlay
▸  Dashboard (top-left corner): Near-Miss Count | Current Risk Level | Vehicles Detected
▸  Bonus: draw a coloured line between two near-miss objects showing their distance

📦  Deliverables — What Your Final System Must Produce

1.  Annotated output video with colour-coded bounding boxes (green / yellow / red)
2.  Track IDs displayed on all detected road users
3.  Risk score or level shown on interacting pairs
4.  Near-miss event counter and visual alert
5.  Written explanation of your risk formula and near-miss definition (in code comments or separate
doc)

Criteria

Engineering Logic

System Design

Functionality

Creativity

Documentation

Weight  What we look for

40%

25%

20%

10%

5%

Risk formula quality, near-miss definition reasoning, factor
justification

Detection → Tracking → Proximity → Risk → Alert pipeline
coherence

System runs on the test video without crashes, detections
look correct

Novel metrics, interesting visualisations, trajectory
prediction

Designer's choices clearly explained in code or written
notes

⭐  Bonus Challenges — Choose One or More

Bonus 1:  Build a live danger heatmap showing which road areas had the most near-miss events.

Bonus 2:  Estimate object speed in km/h using pixel displacement, video FPS, and an assumed
camera height.

Bonus 3:  Predict future collision risk: extend current velocity vectors forward and flag predicted
intersections.

Bonus 4:  Generate a Safety Report at the end: intersection risk score, most dangerous time period,
most involved object types.

