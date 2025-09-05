# Cleaning Buttons Analysis - Myr-Ag User Interface

## Overview of Cleaning Buttons

The Gradio user interface has **6 distinct maintenance buttons**, each with a specific function and different consequences.

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

### 2. 🔨 **Rebuild LEANN Index**
```mermaid
graph LR
    A["Rebuild LEANN Index Button"] --> B["Lit processed/"]
    B --> C["Recrée index LEANN"]
    C --> D["Conserve processed/ et uploads/"]
    
    style A fill:#00b894,stroke:#00a085,stroke-width:2px
    style B fill:#fdcb6e,stroke:#e17055,stroke-width:2px
    style C fill:#74b9ff,stroke:#0984e3,stroke-width:2px
    style D fill:#55a3ff,stroke:#00b894,stroke-width:2px
```

**Function:** Rebuilds the LEANN index from existing processed documents

**Impact:**
- ✅ **Reads:** Processed documents from data/processed/
- ✅ **Rebuilds:** LEANN index with current documents
- ✅ **Preserves:** All processed documents and uploads
- ✅ **Result:** Fresh index with all available documents

**API Code:** `POST /system/rebuild-index`
**Implementation:** ✅ Complete and functional

---

### 3. 🔄 **Reset LlamaIndex Excel**
```mermaid
graph LR
    A["Reset LlamaIndex Excel Button"] --> B["Supprime llamaindex_excel_index/"]
    B --> C["Recrée index LlamaIndex Excel"]
    C --> D["Conserve processed/ et uploads/"]
    
    style A fill:#ff6b6b,stroke:#d63031,stroke-width:2px
    style B fill:#fdcb6e,stroke:#e17055,stroke-width:2px
    style C fill:#74b9ff,stroke:#0984e3,stroke-width:2px
    style D fill:#55a3ff,stroke:#00b894,stroke-width:2px
```

**Function:** Resets only the LlamaIndex Excel database

**Impact:**
- ✅ **Removes:** LlamaIndex Excel index (embeddings, metadata)
- ✅ **Preserves:** Processed Excel files and uploads
- ✅ **Result:** Excel files available but not indexed

**API Code:** `DELETE /system/reset-llamaindex`
**Implementation:** ✅ Complete and functional

---

### 4. 🔨 **Rebuild LlamaIndex Excel**
```mermaid
graph LR
    A["Rebuild LlamaIndex Excel Button"] --> B["Lit processed Excel files"]
    B --> C["Recrée index LlamaIndex Excel"]
    C --> D["Conserve processed/ et uploads/"]
    
    style A fill:#00b894,stroke:#00a085,stroke-width:2px
    style B fill:#fdcb6e,stroke:#e17055,stroke-width:2px
    style C fill:#74b9ff,stroke:#0984e3,stroke-width:2px
    style D fill:#55a3ff,stroke:#00b894,stroke-width:2px
```

**Function:** Rebuilds the LlamaIndex Excel index from existing processed Excel files

**Impact:**
- ✅ **Reads:** Processed Excel files from data/processed/
- ✅ **Rebuilds:** LlamaIndex Excel index with current files
- ✅ **Preserves:** All processed documents and uploads
- ✅ **Result:** Fresh Excel index with all available files

**API Code:** `POST /system/rebuild-llamaindex`
**Implementation:** ✅ Complete and functional

---

### 5. 🗑️ **Clear LEANN Documents**
```mermaid
graph LR
    A["Clear LEANN Documents Button"] --> B["Supprime .leann/"]
    B --> C["Supprime processed/ (non-Excel)"]
    C --> D["Conserve uploads/ et Excel processed/"]
    D --> E["Recrée répertoires vides"]
    
    style A fill:#ff7675,stroke:#d63031,stroke-width:2px
    style B fill:#fdcb6e,stroke:#e17055,stroke-width:2px
    style C fill:#fd79a8,stroke:#e84393,stroke-width:2px
    style D fill:#55a3ff,stroke:#00b894,stroke-width:2px
    style E fill:#74b9ff,stroke:#0984e3,stroke-width:2px
```

**Function:** Clears LEANN index and non-Excel processed documents

**Impact:**
- ✅ **Removes:** LEANN index + Non-Excel processed documents
- ✅ **Preserves:** Upload documents + Excel processed files
- ✅ **Result:** Raw documents + Excel files available, LEANN cleared

**API Code:** `DELETE /system/clear-documents`
**Implementation:** ✅ Complete and functional

---

### 6. 🗑️ **Clear LlamaIndex Excel**
```mermaid
graph LR
    A["Clear LlamaIndex Excel Button"] --> B["Supprime llamaindex_excel_index/"]
    B --> C["Supprime processed/ Excel files"]
    C --> D["Conserve uploads/ et autres processed/"]
    D --> E["Recrée répertoires vides"]
    
    style A fill:#ff7675,stroke:#d63031,stroke-width:2px
    style B fill:#fdcb6e,stroke:#e17055,stroke-width:2px
    style C fill:#fd79a8,stroke:#e84393,stroke-width:2px
    style D fill:#55a3ff,stroke:#00b894,stroke-width:2px
    style E fill:#74b9ff,stroke:#0984e3,stroke-width:2px
```

**Function:** Clears LlamaIndex Excel index and Excel processed files

**Impact:**
- ✅ **Removes:** LlamaIndex Excel index + Excel processed files
- ✅ **Preserves:** Upload documents + Non-Excel processed files
- ✅ **Result:** Raw documents + non-Excel files available, Excel cleared

**API Code:** `DELETE /system/clear-llamaindex`
**Implementation:** ✅ Complete and functional

---

### 7. 💥 **Clear Everything**
```mermaid
graph LR
    A["Clear Everything Button"] --> B["Supprime .leann/"]
    B --> C["Supprime llamaindex_excel_index/"]
    C --> D["Supprime processed/"]
    D --> E["Supprime uploads/"]
    E --> F["Recrée tous les répertoires vides"]
    
    style A fill:#e84393,stroke:#6c5ce7,stroke-width:3px
    style B fill:#fdcb6e,stroke:#e17055,stroke-width:2px
    style C fill:#fd79a8,stroke:#e84393,stroke-width:2px
    style D fill:#ff7675,stroke:#d63031,stroke-width:2px
    style E fill:#ff6b6b,stroke:#d63031,stroke-width:2px
    style F fill:#74b9ff,stroke:#0984e3,stroke-width:2px
```

**Function:** Clears EVERYTHING in the system

**Impact:**
- ✅ **Removes:** LEANN index + LlamaIndex Excel + All processed documents + All upload documents
- ✅ **Preserves:** Nothing
- ✅ **Result:** Completely empty system

**API Code:** `DELETE /system/clear-all`
**Implementation:** ✅ Complete and functional

## Summary

The system provides **7 maintenance buttons** organized by function:

### **Reset Operations** (Index only)
- **Reset LEANN Index** - Clears LEANN index, keeps documents
- **Reset LlamaIndex Excel** - Clears Excel index, keeps files

### **Rebuild Operations** (Recreate from existing)
- **Rebuild LEANN Index** - Rebuilds LEANN from processed documents
- **Rebuild LlamaIndex Excel** - Rebuilds Excel index from processed files

### **Clear Operations** (Remove data)
- **Clear LEANN Documents** - Removes LEANN + non-Excel processed files
- **Clear LlamaIndex Excel** - Removes Excel index + Excel processed files
- **Clear Everything** - Removes everything

Each button requires confirmation via checkbox for safety.



