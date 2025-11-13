# Diagramme de Relations - Smart Data Models pour la Mobilité Urbaine

Ce document contient un diagramme Mermaid illustrant les relations entre les principaux modèles de données pour la gestion du trafic et le stationnement en ville.

```mermaid
classDiagram
    class TrafficFlowObserved {
        +refRoadSegment: RoadSegment
    }
    class RoadSegment {
        +refRoad: Road
    }
    class Road {
        +refRoadSegment: RoadSegment[]
    }
    class OnStreetParking {
        +refParkingSpot: ParkingSpot[]
        +refParkingGroup: ParkingGroup[]
    }
    class ParkingSpot {
        +refParkingSite: OnStreetParking
        +refParkingGroup: ParkingGroup
    }
    class ParkingGroup

    TrafficFlowObserved --|> RoadSegment : refRoadSegment
    RoadSegment --|> Road : refRoad
    Road "1" -- "0..*" RoadSegment : refRoadSegment
    OnStreetParking "1" -- "0..*" ParkingSpot : refParkingSpot
    OnStreetParking "1" -- "0..*" ParkingGroup : refParkingGroup
    ParkingSpot --|> OnStreetParking : refParkingSite
    ParkingSpot --|> ParkingGroup : refParkingGroup
