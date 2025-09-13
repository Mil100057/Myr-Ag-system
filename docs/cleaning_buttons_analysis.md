# System Management Interface Analysis - Myr-Ag User Interface

## Overview of System Management

The Gradio user interface now features a **comprehensive System Management tab** with domain-specific controls and unified operations.

## Interface Structure

### 1. Index Management Section

#### Domain Selection Radio

```mermaid
graph LR
    A["Domain Selection Radio"] --> B["General Index"]
    A --> C["Financial Domain"]
    A --> D["Legal Domain"]
    A --> E["Medical Domain"]
    A --> F["Academic Domain"]
    A --> G["Excel Index"]
    A --> H["All Indexes"]
  
    style A fill:#74b9ff,stroke:#0984e3,stroke-width:2px
    style B fill:#55a3ff,stroke:#00b894,stroke-width:2px
    style C fill:#fdcb6e,stroke:#e17055,stroke-width:2px
    style D fill:#fd79a8,stroke:#e84393,stroke-width:2px
    style E fill:#6c5ce7,stroke:#5f3dc4,stroke-width:2px
    style F fill:#a29bfe,stroke:#6c5ce7,stroke-width:2px
    style G fill:#00b894,stroke:#00a085,stroke-width:2px
    style H fill:#ff7675,stroke:#d63031,stroke-width:2px
```

**Available Options:**

- **General Index**: Manages the main LEANN index
- **Financial Domain**: Manages financial-specific index
- **Legal Domain**: Manages legal-specific index
- **Medical Domain**: Manages medical-specific index
- **Academic Domain**: Manages academic-specific index
- **Excel Index**: Manages LlamaIndex Excel database
- **All Indexes**: Manages all indexes simultaneously

#### Operations

- **Reset Selected Index**: Resets the selected domain index (preserves data)
- **Rebuild Selected Index**: Rebuilds the selected domain index from processed documents
- **Dynamic Description**: Shows description based on selected domain

### 2. Data Management Section

#### Domain Selection Radio

```mermaid
graph LR
    A["Domain Selection Radio"] --> B["General Index"]
    A --> C["Financial Domain"]
    A --> D["Legal Domain"]
    A --> E["Medical Domain"]
    A --> F["Academic Domain"]
    A --> G["Excel Index"]
    A --> H["All Data"]
  
    style A fill:#74b9ff,stroke:#0984e3,stroke-width:2px
    style B fill:#55a3ff,stroke:#00b894,stroke-width:2px
    style C fill:#fdcb6e,stroke:#e17055,stroke-width:2px
    style D fill:#fd79a8,stroke:#e84393,stroke-width:2px
    style E fill:#6c5ce7,stroke:#5f3dc4,stroke-width:2px
    style F fill:#a29bfe,stroke:#6c5ce7,stroke-width:2px
    style G fill:#00b894,stroke:#00a085,stroke-width:2px
    style H fill:#ff7675,stroke:#d63031,stroke-width:2px
```

#### Operations

- **Clear Selected Domain**: Clears the selected domain data (destructive operation)
- **Confirmation Required**: Checkbox to confirm destructive operations
- **Dynamic Description**: Shows impact description based on selection

## API Endpoints

### Index Management Endpoints

| Endpoint                       | Method | Function                      | Domain Support                      |
| ------------------------------ | ------ | ----------------------------- | ----------------------------------- |
| `/system/reset-index`        | POST   | Reset general LEANN index     | General only                        |
| `/domains/{domain}/reset`    | POST   | Reset specific domain index   | Financial, Legal, Medical, Academic |
| `/system/reset-llamaindex`   | POST   | Reset Excel index             | Excel only                          |
| `/system/rebuild-index`      | POST   | Rebuild general LEANN index   | General only                        |
| `/domains/{domain}/rebuild`  | POST   | Rebuild specific domain index | Financial, Legal, Medical, Academic |
| `/system/rebuild-llamaindex` | POST   | Rebuild Excel index           | Excel only                          |
| `/system/rebuild-all`        | POST   | Rebuild all indexes           | All domains                         |

### Data Management Endpoints

| Endpoint                          | Method | Function                          | Domain Support                      |
| --------------------------------- | ------ | --------------------------------- | ----------------------------------- |
| `/system/clear-general`         | DELETE | Clear general index + documents   | General only                        |
| `/system/clear-domain/{domain}` | DELETE | Clear specific domain + documents | Financial, Legal, Medical, Academic |
| `/system/clear-excel`           | DELETE | Clear Excel index + files         | Excel only                          |
| `/system/clear-all`             | DELETE | Clear everything                  | All domains                         |

## Domain-Specific Operations

### General Index Operations

```mermaid
graph TB
    A["General Index Selected"] --> B["Reset General Index"]
    A --> C["Rebuild General Index"]
    A --> D["Clear General Data"]
  
    B --> E["Removes .leann_main_collection"]
    C --> F["Rebuilds from all processed documents"]
    D --> G["Removes index + all processed documents"]
  
    style A fill:#55a3ff,stroke:#00b894,stroke-width:2px
    style B fill:#74b9ff,stroke:#0984e3,stroke-width:2px
    style C fill:#00b894,stroke:#00a085,stroke-width:2px
    style D fill:#ff7675,stroke:#d63031,stroke-width:2px
```

### Domain-Specific Operations (Financial, Legal, Medical, Academic)

```mermaid
graph TB
    A["Domain Selected"] --> B["Reset Domain Index"]
    A --> C["Rebuild Domain Index"]
    A --> D["Clear Domain Data"]
  
    B --> E["Removes .leann_{domain}_collection"]
    C --> F["Rebuilds from domain-specific documents"]
    D --> G["Removes domain index + domain documents"]
  
    style A fill:#fdcb6e,stroke:#e17055,stroke-width:2px
    style B fill:#74b9ff,stroke:#0984e3,stroke-width:2px
    style C fill:#00b894,stroke:#00a085,stroke-width:2px
    style D fill:#ff7675,stroke:#d63031,stroke-width:2px
```

### Excel Index Operations

```mermaid
graph TB
    A["Excel Index Selected"] --> B["Reset Excel Index"]
    A --> C["Rebuild Excel Index"]
    A --> D["Clear Excel Data"]
  
    B --> E["Removes LlamaIndex Excel index"]
    C --> F["Rebuilds from Excel processed files"]
    D --> G["Removes Excel index + Excel files"]
  
    style A fill:#00b894,stroke:#00a085,stroke-width:2px
    style B fill:#74b9ff,stroke:#0984e3,stroke-width:2px
    style C fill:#00b894,stroke:#00a085,stroke-width:2px
    style D fill:#ff7675,stroke:#d63031,stroke-width:2px
```

### All Domains Operations

```mermaid
graph TB
    A["All Domains Selected"] --> B["Reset All Indexes"]
    A --> C["Rebuild All Indexes"]
    A --> D["Clear All Data"]
  
    B --> E["Resets all domain indexes"]
    C --> F["Rebuilds all indexes from processed documents"]
    D --> G["Removes everything (⚠️ DANGER)"]
  
    style A fill:#ff7675,stroke:#d63031,stroke-width:2px
    style B fill:#74b9ff,stroke:#0984e3,stroke-width:2px
    style C fill:#00b894,stroke:#00a085,stroke-width:2px
    style D fill:#ff7675,stroke:#d63031,stroke-width:2px
```

## Safety Features

### Confirmation System

- **Index Operations**: No confirmation required (non-destructive)
- **Data Operations**: Confirmation checkbox required (destructive)
- **All Data Operation**: Extra warning in description

### Operation Impact Matrix

| Operation       | Index Impact | Data Impact | Uploads Impact | Confirmation |
| --------------- | ------------ | ----------- | -------------- | ------------ |
| Reset General   | Removes      | Preserves   | Preserves      | No           |
| Reset Domain    | Removes      | Preserves   | Preserves      | No           |
| Reset Excel     | Removes      | Preserves   | Preserves      | No           |
| Rebuild General | Rebuilds     | Preserves   | Preserves      | No           |
| Rebuild Domain  | Rebuilds     | Preserves   | Preserves      | No           |
| Rebuild Excel   | Rebuilds     | Preserves   | Preserves      | No           |
| Clear General   | Removes      | Removes     | Preserves      | Yes          |
| Clear Domain    | Removes      | Removes     | Preserves      | Yes          |
| Clear Excel     | Removes      | Removes     | Preserves      | Yes          |
| Clear All       | Removes      | Removes     | Removes        | Yes          |

---

*This analysis reflects the current System Management interface as of the latest update with domain-specific controls and unified operations.*
