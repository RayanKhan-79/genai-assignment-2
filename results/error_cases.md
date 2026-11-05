# Error analysis

Stratified misclassified examples (automatic dump).

## extra_evidence (8 cases)

### annotation 306 · sigir-201417 · inclusion
- criterion: Patients with liver space occupying lesion (LOE) that require extended hepatic resection.
- gold: **not enough information** [3] · pred: **not enough information** [2, 3]
- raw output: `{"label": "not enough information", "evidence_ids": [2, 3]}`
- gold sentences:
  - [3] Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.
- predicted sentences:
  - [2] Abdominal ultrasound shows hepatomegaly and abundant free intraperitoneal fluid.
  - [3] Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.

### annotation 307 · sigir-201417 · inclusion
- criterion: Patient selection should be cautious, covering basically 5 types of liver damage which must be submitted prior to liver volumetry: Metastatic Disease subsidiary right hepatectomy extended to segment IV Metastatic Disease subsidiary right hepatectomy with suspected diseased liver (neoadjuvant chemotherapy) (in cases of doubt may be used liver function test indocyanine green) Bilobar liver metastases with multiple nodules in the right lobe and more than 3 nodules greater than 30 mm in the left hepatic lobe (LHI) will perform lumpectomies the LHI + right portal branch ligation (or postoperative percutaneous embolization) in order to make right hepatectomy 4-6 weeks (two stage surgery) Subsidiary Hepatocarcinoma extended right hepatectomy Liver Injury benign / malignant (Hemangiomas, hydatid cysts or liver tumors / primary bile hepatoblastoma), which by extension threatens the viability of the remaining liver tissue.
- gold: **not enough information** [3] · pred: **not enough information** [0, 1, 2, 3, 4]
- raw output: `{"label": "not enough information", "evidence_ids": [0, 1, 2, 3, 4]}`
- gold sentences:
  - [3] Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.
- predicted sentences:
  - [0] A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.
  - [1] The physical examination is remarkable for tenderness and positive Murphy sign.
  - [2] Abdominal ultrasound shows hepatomegaly and abundant free intraperitoneal fluid.
  - [3] Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.
  - [4] After surgery, the patient is taken to the ICU.

### annotation 381 · sigir-201421 · inclusion
- criterion: Age > 18 years at the time of signing the informed consent form
- gold: **included** [0] · pred: **included** [4]
- raw output: `{"label": "included", "evidence_ids": [4]}`
- gold sentences:
  - [0] A 21-year-old female is evaluated for progressive arthralgias and malaise.
- predicted sentences:
  - [4] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.

### annotation 383 · sigir-201421 · inclusion
- criterion: Present with active SLE disease with at least one of the following symptoms: i) Arthritis - > 2 joints with pain and signs of inflammation (i.e. tenderness, swelling, or effusion) ii) Inflammatory-type skin rash iii) Oral ulcers
- gold: **included** [0, 1] · pred: **included** [0, 1, 2]
- raw output: `{"label": "included", "evidence_ids": [0, 1, 2]}`
- gold sentences:
  - [0] A 21-year-old female is evaluated for progressive arthralgias and malaise.
  - [1] On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, a delicate non-palpable purpura on her calves, and swelling and tenderness of her wrists and ankles.
- predicted sentences:
  - [0] A 21-year-old female is evaluated for progressive arthralgias and malaise.
  - [1] On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, a delicate non-palpable purpura on her calves, and swelling and tenderness of her wrists and ankles.
  - [2] Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA.

## missing_evidence (16 cases)

### annotation 179 · sigir-201410 · exclusion
- criterion: Had been performed central venous puncture or catheterization before.
- gold: **not excluded** [0] · pred: **not excluded** []
- raw output: `{"label": "not excluded", "evidence_ids": []}`
- gold sentences:
  - [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
- predicted sentences:

### annotation 196 · sigir-201411 · exclusion
- criterion: Less than 18 years of age
- gold: **not excluded** [0] · pred: **not excluded** []
- raw output: `{"label": "not excluded", "evidence_ids": []}`
- gold sentences:
  - [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
- predicted sentences:

### annotation 197 · sigir-201411 · exclusion
- criterion: Decreased level of consciousness
- gold: **not excluded** [0, 5] · pred: **not excluded** []
- raw output: `{"label": "not excluded", "evidence_ids": []}`
- gold sentences:
  - [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
  - [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- predicted sentences:

### annotation 198 · sigir-201411 · exclusion
- criterion: Inability to answer questions
- gold: **not excluded** [5] · pred: **not excluded** []
- raw output: `{"label": "not excluded", "evidence_ids": []}`
- gold sentences:
  - [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- predicted sentences:

## not_applicable_vs_not_enough_information (3 cases)

### annotation 422 · sigir-201422 · inclusion
- criterion: If female, using birth control
- gold: **not applicable** [3] · pred: **not enough information** [0, 3]
- raw output: `{"label": "not enough information", "evidence_ids": [0, 3]}`
- gold sentences:
  - [3] She has had no sexual partners and her menses are regular.
- predicted sentences:
  - [0] A 15-year-old girl presents to the ER with abdominal pain.
  - [3] She has had no sexual partners and her menses are regular.

### annotation 745 · sigir-201513 · inclusion
- criterion: female patients have adequate birth-control method
- gold: **not applicable** [] · pred: **not enough information** []
- raw output: `{"label": "not enough information", "evidence_ids": []}`
- gold sentences:
- predicted sentences:

### annotation 834 · sigir-201516 · inclusion
- criterion: --Negative pregnancy test for females
- gold: **not applicable** [] · pred: **not enough information** []
- raw output: `{"label": "not enough information", "evidence_ids": []}`
- gold sentences:
- predicted sentences:

## unsupported_evidence (9 cases)

### annotation 166 · sigir-201410 · exclusion
- criterion: Significant upper limb peripheral arterial disease
- gold: **not excluded** [] · pred: **not excluded** [1, 2]
- raw output: `{"label": "not excluded", "evidence_ids": [1, 2]}`
- gold sentences:
- predicted sentences:
  - [1] She is now complaining of a cool right foot.
  - [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.

### annotation 310 · sigir-201417 · exclusion
- criterion: Patients with uncontrolled hypertension.
- gold: **not excluded** [] · pred: **not excluded** [0]
- raw output: `{"label": "not excluded", "evidence_ids": [0]}`
- gold sentences:
- predicted sentences:
  - [0] A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.

### annotation 325 · sigir-201417 · inclusion
- criterion: Genetic patients
- gold: **not enough information** [] · pred: **not enough information** [0]
- raw output: `{"label": "not enough information", "evidence_ids": [0]}`
- gold sentences:
- predicted sentences:
  - [0] A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.

### annotation 384 · sigir-201421 · inclusion
- criterion: Laboratory values as follows Hemoglobin ≥ 100 g/L Absolute neutrophil count ≥ 1.0 x 109/L Total bilirubin ≤ 1.5 x upper limit of normal (ULN) AST (SGOT) / ALT (SGPT) ≤ 2.5 x ULN
- gold: **not enough information** [] · pred: **not enough information** [2]
- raw output: `{"label": "not enough information", "evidence_ids": [2]}`
- gold sentences:
- predicted sentences:
  - [2] Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA.

## wrong_label (56 cases)

### annotation 161 · sigir-201410 · inclusion
- criterion: Patients undergoing elective carotid endarterectomy
- gold: **not included** [] · pred: **not enough information** []
- raw output: `{"label": "not enough information", "evidence_ids": []}`
- gold sentences:
- predicted sentences:

### annotation 162 · sigir-201410 · inclusion
- criterion: Patients undergoing open abdominal aortic aneurysm repair
- gold: **not included** [] · pred: **not enough information** []
- raw output: `{"label": "not enough information", "evidence_ids": []}`
- gold sentences:
- predicted sentences:

### annotation 163 · sigir-201410 · inclusion
- criterion: Patients undergoing endovascular abdominal aneurysm repair
- gold: **not included** [] · pred: **not enough information** [0, 1, 2]
- raw output: `{"label": "not enough information", "evidence_ids": [0, 1, 2]}`
- gold sentences:
- predicted sentences:
  - [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
  - [1] She is now complaining of a cool right foot.
  - [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.

### annotation 164 · sigir-201410 · inclusion
- criterion: Patients undergoing surgical lower limb revascularisation (suprainguinal or infrainguinal)
- gold: **not included** [] · pred: **not enough information** [0]
- raw output: `{"label": "not enough information", "evidence_ids": [0]}`
- gold sentences:
- predicted sentences:
  - [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.

