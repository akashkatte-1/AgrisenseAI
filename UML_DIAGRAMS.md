# AgriSense AI - UML Diagrams

This document contains the UML diagrams for the AgriSense AI application, represented in Mermaid.js syntax. You can view these diagrams using a Mermaid viewer or by rendering this markdown file on GitHub.

## 1. Use Case Diagram
Describes the interactions between actors (Farmer, Admin) and the system.

```mermaid
useCaseDiagram
    actor "Farmer" as F
    actor "Admin" as A
    package "AgriSense AI System" {
        usecase "Register" as UC1
        usecase "Login" as UC2
        usecase "Logout" as UC3
        usecase "View Dashboard" as UC4
        usecase "Get Crop Recommendation" as UC5
        usecase "Get Fertilizer Recommendation" as UC6
        usecase "Detect Plant Disease" as UC7
        usecase "View History" as UC8
        usecase "Manage Users" as UC9
        usecase "View System Stats" as UC10
    }
    F --> UC1
    F --> UC2
    F --> UC3
    F --> UC4
    F --> UC5
    F --> UC6
    F --> UC7
    F --> UC8
    A --> UC2
    A --> UC3
    A --> UC9
    A --> UC10
    A --> UC4
    A --> UC8
```

## 2. Class Diagram
Shows the static structure of the system, including classes, attributes, and relationships.

```mermaid
classDiagram
    class User {
        +ObjectId _id
        +String name
        +String email
        +String password
        +Boolean is_admin
        +DateTime created_at
        +login()
        +logout()
        +register()
    }
    class CropPrediction {
        +ObjectId _id
        +String user_id
        +Integer nitrogen
        +Integer phosphorous
        +Integer potassium
        +Float ph
        +Float rainfall
        +String city
        +String prediction
        +save()
    }
    class FertilizerPrediction {
        +ObjectId _id
        +String user_id
        +String crop_name
        +Integer nitrogen
        +Integer phosphorous
        +Integer potassium
        +String recommendation
        +save()
    }
    class DiseasePrediction {
        +ObjectId _id
        +String user_id
        +String image_filename
        +String disease_name
        +save()
    }
    class App {
        +route_home()
        +route_login()
        +route_crop_predict()
        +route_fertilizer_predict()
        +route_disease_predict()
    }
    
    User "1" -- "*" CropPrediction : makes
    User "1" -- "*" FertilizerPrediction : makes
    User "1" -- "*" DiseasePrediction : makes
    App ..> User : manages
    App ..> CropPrediction : creates
```

## 3. Sequence Diagram
Illustrates the flow of interactions for the "Crop Recommendation" scenario.

```mermaid
sequenceDiagram
    actor User
    participant Frontend as Web Interface
    participant Backend as Flask Server
    participant WeatherAPI as OpenWeatherMap
    participant Model as ML Model
    participant DB as MongoDB

    User->>Frontend: Enter Soil Data & City
    Frontend->>Backend: POST /crop-predict
    Backend->>WeatherAPI: Get Weather(City)
    WeatherAPI-->>Backend: Return Temp, Humidity
    Backend->>Model: Predict(Soil, Weather)
    Model-->>Backend: Return Crop Name
    Backend->>DB: Save Prediction
    DB-->>Backend: Confirm Save
    Backend-->>Frontend: Display Result
    Frontend-->>User: Show Recommended Crop
```

## 4. Collaboration Diagram
Shows the structural organization of objects that send and receive messages.

```mermaid
graph TD
    User((User)) -- 1. Submit Data --> UI[Web Interface]
    UI -- 2. Send Request --> Server[Flask Server]
    Server -- 3. Fetch Weather --> API[Weather API]
    API -- 4. Return Data --> Server
    Server -- 5. Invoke --> Model[ML Model]
    Model -- 6. Return Result --> Server
    Server -- 7. Store Data --> DB[MongoDB]
    Server -- 8. Return Response --> UI
    UI -- 9. Show Result --> User
```

## 5. Activity Diagram
Depicts the workflow for the "Disease Detection" process.

```mermaid
flowchart TD
    A([Start]) --> B{User Logged In?}
    B -- No --> C[Login]
    C --> B
    B -- Yes --> D[Navigate to Disease Detection]
    D --> E[Upload Leaf Image]
    E --> F{Is Image Valid?}
    F -- No --> G[Show Error]
    G --> E
    F -- Yes --> H[Preprocess Image]
    H --> I[Pass to PyTorch Model]
    I --> J[Get Prediction]
    J --> K[Save to Database]
    K --> L[Display Result & Cure]
    L --> M([End])
```

## 6. State Chart Diagram
Shows the states of a `PredictionRequest` object during its lifecycle.

```mermaid
stateDiagram-v2
    [*] --> Initiated
    Initiated --> Processing : Data Received
    Processing --> ExternalAPICall : Need Weather Data
    ExternalAPICall --> Processing : Data Received
    Processing --> ModelInference : Input Prepared
    ModelInference --> ResultGenerated : Prediction Made
    ResultGenerated --> Stored : Saved to DB
    Stored --> [*]
    
    Processing --> Error : Invalid Data
    Error --> [*]
```

## 7. Component Diagram
Illustrates the physical components of the system and their dependencies.

```mermaid
graph TB
    subgraph Client_Side
        Browser[Web Browser]
    end
    
    subgraph Server_Side
        Flask[Flask App]
        Auth[Auth Module]
        Routes[API Routes]
        MLEngine[ML Engine]
    end
    
    subgraph Data_Layer
        MongoDB[(MongoDB Atlas)]
    end
    
    subgraph External_Services
        WeatherAPI[OpenWeatherMap API]
    end
    
    Browser -- HTTP Request --> Flask
    Flask -- Uses --> Auth
    Flask -- Uses --> Routes
    Routes -- Calls --> MLEngine
    Routes -- Queries --> MongoDB
    Routes -- Calls --> WeatherAPI
```

## 8. Deployment Diagram
Shows the hardware nodes and the execution environments.

```mermaid
graph TD
    node1[User Device]
    node2[Web Server Host]
    node3[Database Server]
    
    node1 -- Internet --> node2
    node2 -- Secure Connection --> node3
    
    subgraph User Device
        Browser[Web Browser]
    end
    
    subgraph Web Server Host
        OS[OS: Linux/Windows]
        Python[Python Runtime]
        App[Flask Application]
        Models[ML Models Files]
    end
    
    subgraph Database Server
        Mongo[MongoDB Atlas Instance]
    end
    
    Browser -.-> App
    App -.-> Mongo
```
