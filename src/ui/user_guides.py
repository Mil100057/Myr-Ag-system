# User Guides for Myr-Ag Frontend
# Multi-language user guides for the Gradio interface

def get_english_guide():
    """Get the English user guide content."""
    return """
    ### Managing Your System
    
    The system provides comprehensive management tools for maintaining your document collections and indexes.
    
    #### **Document Management**
    
    **View Documents:**
    * **Document List**: See all uploaded and processed documents
    * **Status Information**: View processing status, chunk counts, and file sizes
    * **Delete Documents**: Remove individual documents from the system
    
    **Upload Options:**
    * **Upload & Process**: Immediately process documents for querying
    * **Upload Only**: Store documents without processing
    * **Process Existing**: Process documents already in uploads
    * **Process Uploaded Only**: Process only unprocessed documents
    * **Index by Domain**: Index documents by domain with automatic detection or manual assignment
    
    #### **System Maintenance**
    
    **Domain Management:**
    * **Domain Selection**: Choose specific domain (General, Financial, Legal, Medical, Academic, Excel)
    * **Domain Indexing**: Index documents by domain with automatic detection or manual assignment
    * **Individual Control**: Manage each domain independently
    * **Bulk Operations**: Reset/rebuild all domains at once
    
    **Index Management (Non-Destructive):**
    * **Reset Selected Index**: Rebuilds selected domain index, preserves all data
    * **Rebuild Selected Index**: Rebuilds from existing processed documents
    * **Domain-Specific**: Each domain has its own isolated vector index
    
    **Data Management (Destructive - Confirmation Required):**
    * **Clear Selected Domain**: Removes selected domain index + domain documents
    * **Clear All Data**: Removes all indexes and all data (⚠️ DANGER)
    * **Safety Features**: Confirmation required for destructive operations
    
    #### **Vector Store Management**
    
    **LEANN Vector Store:**
    * **Comprehensive Monitoring**: View all 7 collections (main + 6 domains) with detailed information
    * **Collection Details**: See individual collection status, document count, and chunk count
    * **Summary Statistics**: View aggregated totals across all collections
    * **Real-time Updates**: Automatic status updates with accurate counts
    * **Domain-Specific Info**: Each domain collection shows its own metrics and status
    
    **LlamaIndex Excel Store:**
    * **Status Monitoring**: View Excel index status and file counts
    * **Excel File Tracking**: Monitor uploaded and processed Excel files
    * **Refresh Status**: Update Excel index information
    
    #### **System Information**
    
    **Model Management:**
    * **Available Models**: View all installed Ollama models
    * **Model Selection**: Choose the best model for your needs
    * **Dynamic Switching**: Change models without restarting the system
    
    **System Status:**
    * **Accurate Statistics**: View correct total documents and chunks across all domains
    * **Real-time Monitoring**: See current system status with live updates
    * **Performance Metrics**: Monitor CPU, memory, and disk usage
    * **System Logs**: Access detailed logs for debugging
    
    **Processing Status:**
    * **Real-time Monitoring**: See current processing status
    * **Queue Information**: View pending processing tasks
    * **Performance Metrics**: Monitor system performance
    
    #### **Best Practices for System Management**
    
    **Domain Organization:**
    * Assign documents to appropriate domains for optimal processing
    * Use domain-specific queries for better results
    * Monitor domain-specific indexes separately
    
    **Regular Maintenance:**
    * Use "Rebuild" operations for quick index refresh
    * Use "Reset" operations when indexes become corrupted
    * Use "Clear" operations only when you want to remove data
    * Manage domains individually or use bulk operations
    
    **Troubleshooting:**
    * If queries return poor results, try rebuilding the relevant domain index
    * If processing fails, check the logs and try reprocessing
    * If memory issues occur, clear unused domain data and rebuild
    * Use domain-specific operations for targeted fixes
    
    **Data Safety:**
    * Always backup important documents before major operations
    * Use "Reset" instead of "Clear" when possible
    * Test operations on small datasets first
    * Use domain-specific operations to minimize impact
    """

def get_french_guide():
    """Get the French user guide content."""
    return """
    ### Gérer votre système
    
    Le système fournit des outils de gestion complets pour maintenir vos collections de documents et index.
    
    #### **Gestion des documents**
    
    **Voir les documents :**
    * **Liste des documents** : Voir tous les documents téléchargés et traités
    * **Informations de statut** : Voir le statut de traitement, le nombre de chunks et la taille des fichiers
    * **Supprimer des documents** : Retirer des documents individuels du système
    
    **Options de téléchargement :**
    * **Télécharger et traiter** : Traiter immédiatement les documents pour les requêtes
    * **Télécharger seulement** : Stocker les documents sans les traiter
    * **Traiter l'existant** : Traiter les documents déjà dans uploads
    * **Traiter les téléchargés seulement** : Traiter uniquement les documents non traités
    * **Indexer par domaine** : Indexer les documents par domaine avec détection automatique ou assignation manuelle
    
    #### **Maintenance du système**
    
    **Gestion des domaines :**
    * **Sélection de domaine** : Choisir un domaine spécifique (Général, Financier, Légal, Médical, Académique, Excel)
    * **Indexation par domaine** : Indexer les documents par domaine avec détection automatique ou assignation manuelle
    * **Contrôle individuel** : Gérer chaque domaine indépendamment
    * **Opérations en masse** : Réinitialiser/reconstruire tous les domaines à la fois
    
    **Gestion des index (Non destructif) :**
    * **Réinitialiser l'index sélectionné** : Reconstruit l'index du domaine sélectionné, préserve toutes les données
    * **Reconstruire l'index sélectionné** : Reconstruit à partir des documents traités existants
    * **Spécifique au domaine** : Chaque domaine a son propre index vectoriel isolé
    
    **Gestion des données (Destructif - Confirmation requise) :**
    * **Effacer le domaine sélectionné** : Supprime l'index du domaine + documents du domaine
    * **Effacer toutes les données** : Supprime tous les index et toutes les données (⚠️ DANGER)
    * **Fonctionnalités de sécurité** : Confirmation requise pour les opérations destructives
    
    #### **Gestion des magasins vectoriels**
    
    **Magasin vectoriel LEANN :**
    * **Surveillance complète** : Voir les 7 collections (principale + 6 domaines) avec informations détaillées
    * **Détails des collections** : Voir le statut, le nombre de documents et de chunks de chaque collection
    * **Statistiques récapitulatives** : Voir les totaux agrégés de toutes les collections
    * **Mises à jour en temps réel** : Actualisations automatiques avec comptages précis
    * **Informations spécifiques au domaine** : Chaque collection de domaine affiche ses propres métriques
    
    **Magasin LlamaIndex Excel :**
    * **Surveillance du statut** : Voir le statut de l'index Excel et le nombre de fichiers
    * **Suivi des fichiers Excel** : Surveiller les fichiers Excel téléchargés et traités
    * **Actualiser le statut** : Mettre à jour les informations de l'index Excel
    
    #### **Informations du système**
    
    **Gestion des modèles :**
    * **Modèles disponibles** : Voir tous les modèles Ollama installés
    * **Sélection de modèle** : Choisir le meilleur modèle pour vos besoins
    * **Changement dynamique** : Changer de modèles sans redémarrer le système
    
    **Statut du système :**
    * **Statistiques précises** : Voir le nombre correct de documents et chunks dans tous les domaines
    * **Surveillance en temps réel** : Voir le statut actuel du système avec mises à jour en direct
    * **Métriques de performance** : Surveiller CPU, mémoire et utilisation du disque
    * **Logs du système** : Accéder aux logs détaillés pour le débogage
    
    **Statut de traitement :**
    * **Surveillance en temps réel** : Voir le statut de traitement actuel
    * **Informations de la file** : Voir les tâches de traitement en attente
    * **Métriques de performance** : Surveiller les performances du système
    
    #### **Meilleures pratiques pour la gestion du système**
    
    **Organisation des domaines :**
    * Assignez les documents aux domaines appropriés pour un traitement optimal
    * Utilisez des requêtes spécifiques au domaine pour de meilleurs résultats
    * Surveillez les index spécifiques au domaine séparément
    
    **Maintenance régulière :**
    * Utilisez les opérations "Reconstruire" pour un rafraîchissement rapide de l'index
    * Utilisez les opérations "Réinitialiser" quand les index deviennent corrompus
    * Utilisez les opérations "Effacer" seulement quand vous voulez supprimer des données
    * Gérez les domaines individuellement ou utilisez les opérations en masse
    
    **Dépannage :**
    * Si les requêtes retournent de mauvais résultats, essayez de reconstruire l'index du domaine pertinent
    * Si le traitement échoue, vérifiez les logs et essayez de retraiter
    * Si des problèmes de mémoire surviennent, effacez les données du domaine inutilisé et reconstruisez
    * Utilisez les opérations spécifiques au domaine pour des corrections ciblées
    
    **Sécurité des données :**
    * Sauvegardez toujours les documents importants avant les opérations majeures
    * Utilisez "Réinitialiser" au lieu de "Effacer" quand possible
    * Testez les opérations sur de petits ensembles de données d'abord
    * Utilisez les opérations spécifiques au domaine pour minimiser l'impact
    """

def get_spanish_guide():
    """Get the Spanish user guide content."""
    return """
    ### Gestionar tu sistema
    
    El sistema proporciona herramientas de gestión completas para mantener tus colecciones de documentos e índices.
    
    #### **Gestión de documentos**
    
    **Ver documentos:**
    * **Lista de documentos** : Ver todos los documentos cargados y procesados
    * **Información de estado** : Ver el estado de procesamiento, número de chunks y tamaño de archivos
    * **Eliminar documentos** : Remover documentos individuales del sistema
    
    **Opciones de carga:**
    * **Cargar y procesar** : Procesar inmediatamente los documentos para consultas
    * **Solo cargar** : Almacenar documentos sin procesarlos
    * **Procesar existente** : Procesar documentos ya en uploads
    * **Procesar solo cargados** : Procesar únicamente documentos no procesados
    * **Indexar por dominio** : Indexar documentos por dominio con detección automática o asignación manual
    
    #### **Mantenimiento del sistema**
    
    **Gestión de dominios:**
    * **Selección de dominio** : Elegir un dominio específico (General, Financiero, Legal, Médico, Académico, Excel)
    * **Indexación por dominio** : Indexar documentos por dominio con detección automática o asignación manual
    * **Control individual** : Gestionar cada dominio independientemente
    * **Operaciones en masa** : Resetear/reconstruir todos los dominios a la vez
    
    **Gestión de índices (No destructivo):**
    * **Resetear índice seleccionado** : Reconstruye el índice del dominio seleccionado, preserva todos los datos
    * **Reconstruir índice seleccionado** : Reconstruye desde documentos procesados existentes
    * **Específico del dominio** : Cada dominio tiene su propio índice vectorial aislado
    
    **Gestión de datos (Destructivo - Confirmación requerida):**
    * **Limpiar dominio seleccionado** : Elimina el índice del dominio + documentos del dominio
    * **Limpiar todos los datos** : Elimina todos los índices y todos los datos (⚠️ PELIGRO)
    * **Características de seguridad** : Confirmación requerida para operaciones destructivas
    
    #### **Gestión de almacenes vectoriales**
    
    **Almacén vectorial LEANN:**
    * **Monitoreo completo** : Ver las 7 colecciones (principal + 6 dominios) con información detallada
    * **Detalles de colecciones** : Ver el estado, número de documentos y chunks de cada colección
    * **Estadísticas resumidas** : Ver totales agregados de todas las colecciones
    * **Actualizaciones en tiempo real** : Actualizaciones automáticas con conteos precisos
    * **Información específica del dominio** : Cada colección de dominio muestra sus propias métricas
    
    **Almacén LlamaIndex Excel:**
    * **Monitoreo de estado** : Ver el estado del índice Excel y número de archivos
    * **Seguimiento de archivos Excel** : Monitorear archivos Excel cargados y procesados
    * **Actualizar estado** : Actualizar información del índice Excel
    
    #### **Información del sistema**
    
    **Gestión de modelos:**
    * **Modelos disponibles** : Ver todos los modelos Ollama instalados
    * **Selección de modelo** : Elegir el mejor modelo para tus necesidades
    * **Cambio dinámico** : Cambiar modelos sin reiniciar el sistema
    
    **Estado del sistema:**
    * **Estadísticas precisas** : Ver el número correcto de documentos y chunks en todos los dominios
    * **Monitoreo en tiempo real** : Ver el estado actual del sistema con actualizaciones en vivo
    * **Métricas de rendimiento** : Monitorear CPU, memoria y uso de disco
    * **Logs del sistema** : Acceder a logs detallados para depuración
    
    **Estado de procesamiento:**
    * **Monitoreo en tiempo real** : Ver el estado de procesamiento actual
    * **Información de cola** : Ver tareas de procesamiento pendientes
    * **Métricas de rendimiento** : Monitorear el rendimiento del sistema
    
    #### **Mejores prácticas para la gestión del sistema**
    
    **Organización de dominios:**
    * Asigna documentos a dominios apropiados para procesamiento óptimo
    * Usa consultas específicas del dominio para mejores resultados
    * Monitorea índices específicos del dominio por separado
    
    **Mantenimiento regular:**
    * Usa operaciones "Reconstruir" para refresco rápido del índice
    * Usa operaciones "Resetear" cuando los índices se corrompan
    * Usa operaciones "Limpiar" solo cuando quieras eliminar datos
    * Gestiona dominios individualmente o usa operaciones en masa
    
    **Solución de problemas:**
    * Si las consultas devuelven malos resultados, intenta reconstruir el índice del dominio relevante
    * Si el procesamiento falla, revisa los logs e intenta reprocesar
    * Si surgen problemas de memoria, limpia datos del dominio no utilizados y reconstruye
    * Usa operaciones específicas del dominio para correcciones dirigidas
    
    **Seguridad de datos:**
    * Siempre respalda documentos importantes antes de operaciones mayores
    * Usa "Resetear" en lugar de "Limpiar" cuando sea posible
    * Prueba operaciones en pequeños conjuntos de datos primero
    * Usa operaciones específicas del dominio para minimizar el impacto
    """

def get_german_guide():
    """Get the German user guide content."""
    return """
    ### Ihr System verwalten
    
    Das System bietet umfassende Verwaltungstools zur Pflege Ihrer Dokumentensammlungen und Indizes.
    
    #### **Dokumentenverwaltung**
    
    **Dokumente anzeigen:**
    * **Dokumentenliste** : Alle hochgeladenen und verarbeiteten Dokumente anzeigen
    * **Statusinformationen** : Verarbeitungsstatus, Chunk-Anzahl und Dateigröße anzeigen
    * **Dokumente löschen** : Einzelne Dokumente aus dem System entfernen
    
    **Upload-Optionen:**
    * **Hochladen und verarbeiten** : Dokumente sofort für Abfragen verarbeiten
    * **Nur hochladen** : Dokumente ohne Verarbeitung speichern
    * **Vorhandene verarbeiten** : Bereits hochgeladene Dokumente verarbeiten
    * **Nur hochgeladene verarbeiten** : Nur unverarbeitete Dokumente verarbeiten
    * **Nach Domäne indexieren** : Dokumente nach Domäne mit automatischer Erkennung oder manueller Zuweisung indexieren
    
    #### **Systemwartung**
    
    **Domänenverwaltung:**
    * **Domänenauswahl** : Spezifische Domäne wählen (Allgemein, Finanziell, Rechtlich, Medizinisch, Akademisch, Excel)
    * **Domänenindexierung** : Dokumente nach Domäne mit automatischer Erkennung oder manueller Zuweisung indexieren
    * **Individuelle Kontrolle** : Jede Domäne unabhängig verwalten
    * **Massenoperationen** : Alle Domänen auf einmal zurücksetzen/neu aufbauen
    
    **Indexverwaltung (Nicht-destruktiv):**
    * **Ausgewählten Index zurücksetzen** : Baut ausgewählten Domänenindex neu auf, bewahrt alle Daten
    * **Ausgewählten Index neu aufbauen** : Baut aus vorhandenen verarbeiteten Dokumenten neu auf
    * **Domänenspezifisch** : Jede Domäne hat ihren eigenen isolierten Vektorindex
    
    **Datenverwaltung (Destruktiv - Bestätigung erforderlich):**
    * **Ausgewählte Domäne löschen** : Entfernt Domänenindex + Domänendokumente
    * **Alle Daten löschen** : Entfernt alle Indizes und alle Daten (⚠️ GEFÄHRLICH)
    * **Sicherheitsfunktionen** : Bestätigung für destruktive Operationen erforderlich
    
    #### **Vektorspeicher-Verwaltung**
    
    **LEANN-Vektorspeicher:**
    * **Umfassende Überwachung** : Alle 7 Sammlungen (Haupt- + 6 Domänen) mit detaillierten Informationen anzeigen
    * **Sammlungsdetails** : Status, Dokumentenanzahl und Chunk-Anzahl jeder Sammlung anzeigen
    * **Zusammenfassende Statistiken** : Aggregierte Gesamtwerte aller Sammlungen anzeigen
    * **Echtzeit-Updates** : Automatische Aktualisierungen mit präzisen Zählungen
    * **Domänenspezifische Informationen** : Jede Domänensammlung zeigt ihre eigenen Metriken
    
    **LlamaIndex Excel-Speicher:**
    * **Statusüberwachung** : Excel-Indexstatus und Dateianzahl anzeigen
    * **Excel-Datei-Tracking** : Hochgeladene und verarbeitete Excel-Dateien überwachen
    * **Status aktualisieren** : Excel-Indexinformationen aktualisieren
    
    #### **Systeminformationen**
    
    **Modellverwaltung:**
    * **Verfügbare Modelle** : Alle installierten Ollama-Modelle anzeigen
    * **Modellauswahl** : Bestes Modell für Ihre Bedürfnisse wählen
    * **Dynamischer Wechsel** : Modelle ohne Systemneustart wechseln
    
    **Systemstatus:**
    * **Präzise Statistiken** : Korrekte Gesamtzahl von Dokumenten und Chunks in allen Domänen anzeigen
    * **Echtzeitüberwachung** : Aktuellen Systemstatus mit Live-Updates anzeigen
    * **Leistungsmetriken** : CPU, Speicher und Festplattennutzung überwachen
    * **Systemlogs** : Detaillierte Logs für Debugging zugreifen
    
    **Verarbeitungsstatus:**
    * **Echtzeitüberwachung** : Aktuellen Verarbeitungsstatus anzeigen
    * **Warteschlangeninformationen** : Ausstehende Verarbeitungsaufgaben anzeigen
    * **Leistungsmetriken** : Systemleistung überwachen
    
    #### **Beste Praktiken für Systemverwaltung**
    
    **Domänenorganisation:**
    * Dokumente zu passenden Domänen zuweisen für optimale Verarbeitung
    * Domänenspezifische Abfragen für bessere Ergebnisse verwenden
    * Domänenspezifische Indizes separat überwachen
    
    **Regelmäßige Wartung:**
    * "Neu aufbauen"-Operationen für schnelle Indexaktualisierung verwenden
    * "Zurücksetzen"-Operationen verwenden, wenn Indizes korrupt werden
    * "Löschen"-Operationen nur verwenden, wenn Sie Daten entfernen möchten
    * Domänen einzeln verwalten oder Massenoperationen verwenden
    
    **Fehlerbehebung:**
    * Wenn Abfragen schlechte Ergebnisse liefern, versuchen Sie, den relevanten Domänenindex neu aufzubauen
    * Wenn die Verarbeitung fehlschlägt, überprüfen Sie die Logs und versuchen Sie eine Neuverarbeitung
    * Bei Speicherproblemen, ungenutzte Domänendaten löschen und neu aufbauen
    * Domänenspezifische Operationen für gezielte Korrekturen verwenden
    
    **Datensicherheit:**
    * Wichtige Dokumente vor größeren Operationen immer sichern
    * "Zurücksetzen" statt "Löschen" verwenden, wenn möglich
    * Operationen zuerst an kleinen Datensätzen testen
    * Domänenspezifische Operationen verwenden, um Auswirkungen zu minimieren
    """
