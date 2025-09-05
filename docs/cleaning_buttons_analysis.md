# Cleaning Buttons Analysis - Myr-Ag User Interface

## Overview of Cleaning Buttons

The Gradio user interface has **3 distinct cleaning buttons**, each with a specific function and different consequences.

## Available Buttons

### 1. 🔄 **Reset LEANN Index**
```mermaid
graph LR
    A["Reset LEANN Index Button"] --> B["Supprime .leann/"]
    B --> C["Recrée index LEANN"]
    C --> D["Conserve processed/ et uploads/"]
    
    style A fill:#ff6b6b,stroke:#d63031,stroke-width:2px
    style B fill:#fdcb6e,stroke:#e17055,stroke-width:2px
    style C fill:#74b9ff,stroke:#0984e3,stroke-width:2px
    style D fill:#55a3ff,stroke:#00b894,stroke-width:2px
```

**Function:** Resets only the LEANN vector database

**Impact:**
- ✅ **Removes:** LEANN index (embeddings, metadata)
- ✅ **Preserves:** Processed documents and uploads
- ✅ **Result:** Documents available but not indexed

**API Code:** `DELETE /system/reset-index`
**Implementation:** ✅ Complete and functional

---

### 2. 🗑️ **Clear Documents**
```mermaid
graph LR
    A["Clear Documents Button"] --> B["Supprime .leann/"]
    B --> C["Supprime processed/"]
    C --> D["Conserve uploads/"]
    D --> E["Recrée répertoires vides"]
    
    style A fill:#ff7675,stroke:#d63031,stroke-width:2px
    style B fill:#fdcb6e,stroke:#e17055,stroke-width:2px
    style C fill:#fd79a8,stroke:#e84393,stroke-width:2px
    style D fill:#55a3ff,stroke:#00b894,stroke-width:2px
    style E fill:#74b9ff,stroke:#0984e3,stroke-width:2px
```

**Function:** Clears processed documents and LEANN index

**Impact:**
- ✅ **Removes:** LEANN index + Processed documents
- ✅ **Preserves:** Upload documents (raw)
- ✅ **Result:** Raw documents available, everything else cleared

**API Code:** `DELETE /system/clear-documents`
**Implementation:** ✅ Complete and functional

---

### 3. 💥 **Clear Everything**
```mermaid
graph LR
    A["Clear Everything Button"] --> B["Supprime .leann/"]
    B --> C["Supprime processed/"]
    C --> D["Supprime uploads/"]
    D --> E["Recrée tous les répertoires vides"]
    
    style A fill:#e84393,stroke:#6c5ce7,stroke-width:3px
    style B fill:#fdcb6e,stroke:#e17055,stroke-width:2px
    style C fill:#fd79a8,stroke:#e84393,stroke-width:2px
    style D fill:#ff7675,stroke:#d63031,stroke-width:2px
    style E fill:#74b9ff,stroke:#0984e3,stroke-width:2px
```

**Function:** Clears EVERYTHING in the system

**Impact:**
- ✅ **Removes:** LEANN index + Processed documents + Upload documents
- ✅ **Preserves:** Nothing
- ✅ **Result:** Completely empty system

**API Code:** `DELETE /system/clear-all`
**Implementation:** ✅ Complete and functional



