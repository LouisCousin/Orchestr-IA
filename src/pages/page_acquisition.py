"""Page d'acquisition du corpus documentaire."""

import streamlit as st
from pathlib import Path

from src.utils.config import ROOT_DIR
from src.utils.file_utils import ensure_dir, save_json, sanitize_filename
from src.utils.token_counter import count_tokens
from src.core.corpus_acquirer import CorpusAcquirer, AcquisitionReport
from src.core.text_extractor import extract
from src.core.corpus_deduplicator import CorpusDeduplicator
from src.core.cost_tracker import CostTracker
from src.utils.content_validator import is_antibot_page


PROJECTS_DIR = ROOT_DIR / "projects"


def render():
    st.title("Acquisition du corpus")
    st.info(
        "**Étape 2/5** — Constituez votre corpus de sources. Uploadez des fichiers "
        "(PDF, DOCX, etc.) ou saisissez des URLs. Ces documents serviront de base "
        "factuelle pour la génération (RAG). Plus le corpus est riche et pertinent, "
        "meilleur sera le document produit."
    )
    st.markdown("---")

    if not st.session_state.project_state:
        st.warning("Aucun projet actif. Créez ou ouvrez un projet depuis la page Accueil.")
        return

    project_id = st.session_state.current_project
    corpus_dir = PROJECTS_DIR / project_id / "corpus"
    ensure_dir(corpus_dir)

    # Sources d'acquisition (côte à côte)
    col_upload, col_urls = st.columns(2)

    with col_upload:
        _render_file_upload(corpus_dir)

    with col_urls:
        _render_url_acquisition(corpus_dir)

    # Récapitulatif du corpus
    st.markdown("---")
    _render_corpus_recap(corpus_dir)

    # Phase 3: GROBID status and metadata overrides
    project_dir = PROJECTS_DIR / project_id
    _render_metadata_overrides(corpus_dir, project_dir)

    # Inspection RAG
    _render_rag_inspection(project_dir)


def _render_file_upload(corpus_dir: Path):
    """Upload de fichiers locaux."""
    st.subheader("Fichiers locaux")

    uploaded_files = st.file_uploader(
        "Sélectionnez vos fichiers sources",
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt", "md", "html"],
        accept_multiple_files=True,
        help="Formats supportés : PDF, DOCX, Excel, CSV, TXT, Markdown, HTML",
    )

    if uploaded_files and st.button("Ajouter au corpus", type="primary"):
        acquirer = CorpusAcquirer(corpus_dir)
        report = AcquisitionReport()

        progress = st.progress(0, text="Acquisition en cours...")
        for i, uploaded in enumerate(uploaded_files):
            # Sauvegarder le fichier temporairement (nom sanitisé)
            safe_name = sanitize_filename(uploaded.name)
            temp_path = corpus_dir / f"_temp_{safe_name}"
            temp_path.write_bytes(uploaded.getvalue())

            acquirer.acquire_local_files([temp_path], report)

            # Supprimer le fichier temporaire
            if temp_path.exists():
                temp_path.unlink()

            progress.progress((i + 1) / len(uploaded_files), text=f"Traitement de {uploaded.name}...")

        progress.empty()
        _display_report(report)
        st.rerun()


def _render_url_acquisition(corpus_dir: Path):
    """Acquisition depuis URLs."""
    st.subheader("URLs distantes")

    urls_text = st.text_area(
        "URLs (une par ligne)",
        height=120,
        placeholder="https://example.com/document.pdf\nhttps://example.com/page-web",
    )

    url_file = st.file_uploader(
        "Ou importer depuis un fichier Excel/CSV",
        type=["xlsx", "csv"],
        key="url_file",
    )

    slow_mode = st.checkbox("Mode sites lents (timeouts étendus)")

    if st.button("Lancer l'acquisition", type="primary"):
        config = st.session_state.config
        acq_config = config.get("corpus_acquisition", {})

        if slow_mode:
            conn_timeout = acq_config.get("slow_mode_connection_timeout", 30)
            read_timeout = acq_config.get("slow_mode_read_timeout", 120)
        else:
            conn_timeout = acq_config.get("connection_timeout", 15)
            read_timeout = acq_config.get("read_timeout", 60)

        acquirer = CorpusAcquirer(
            corpus_dir=corpus_dir,
            connection_timeout=conn_timeout,
            read_timeout=read_timeout,
            throttle_delay=acq_config.get("throttle_delay", 1.0),
            user_agent=acq_config.get("user_agent", "Mozilla/5.0"),
        )

        report = AcquisitionReport()

        # URLs depuis le fichier
        if url_file:
            temp_path = corpus_dir / f"_temp_{sanitize_filename(url_file.name)}"
            temp_path.write_bytes(url_file.getvalue())
            try:
                acquirer.acquire_urls_from_file(temp_path, report)
            finally:
                temp_path.unlink(missing_ok=True)

        # URLs saisies
        if urls_text.strip():
            urls = [u.strip() for u in urls_text.strip().split("\n") if u.strip()]
            if urls:
                with st.spinner(f"Téléchargement de {len(urls)} URL(s)..."):
                    acquirer.acquire_urls(urls, report)

        acquirer.close()
        _display_report(report)
        st.rerun()


def _render_corpus_recap(corpus_dir: Path):
    """Récapitulatif du corpus acquis avec pré-analyse."""
    st.subheader("Corpus acquis")

    files = sorted(f for f in corpus_dir.iterdir() if f.is_file() and not f.name.startswith(".") and f.suffix != ".json")

    if not files:
        st.info("Aucun document dans le corpus. Ajoutez des fichiers ou URLs ci-dessus.")
        return

    # Analyse du corpus
    documents_info = []
    deduplicator = CorpusDeduplicator(corpus_dir)
    extractions = []

    with st.spinner("Analyse du corpus..."):
        for f in files:
            result = extract(f)
            extractions.append(result)

            tokens = count_tokens(result.text) if result.text else 0
            dedup_entry = deduplicator.check_duplicate(result)

            # Évaluation de la qualité du contenu
            quality = "OK"
            if result.text:
                if is_antibot_page(result.text):
                    quality = "Suspect"
                elif len(result.text.strip()) < 200:
                    quality = "Suspect"
            elif not result.text:
                quality = "Vide"

            documents_info.append({
                "Fichier": f.name,
                "Pages": result.page_count,
                "Tokens": tokens,
                "Qualité": quality,
                "Statut extraction": result.status,
                "Méthode": result.extraction_method,
                "Taille (Ko)": round(result.source_size_bytes / 1024, 1),
                "Doublon": dedup_entry.status,
            })

            if dedup_entry.status == "unique":
                deduplicator.register(dedup_entry)

    # Tableau récapitulatif
    import pandas as pd
    df = pd.DataFrame(documents_info)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Statistiques agrégées
    total_tokens = sum(d["Tokens"] for d in documents_info)
    total_files = len(documents_info)
    duplicates = sum(1 for d in documents_info if d["Doublon"] != "unique")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", total_files)
    col2.metric("Tokens totaux", f"{total_tokens:,}")
    col3.metric("Doublons détectés", duplicates)

    # Estimation de coûts
    config = st.session_state.project_state.config
    model = config.get("model", "gpt-4o")
    provider = config.get("default_provider", "openai")
    tracker = CostTracker()

    docs_for_estimate = [{"tokens": d["Tokens"]} for d in documents_info]
    estimate = tracker.estimate_corpus_cost(docs_for_estimate, provider, model)
    if "error" not in estimate:
        col4.metric("Coût input estimé", f"${estimate['estimated_input_cost_usd']:.4f}")

        if estimate.get("documents_exceeding_context", 0) > 0:
            st.warning(
                f"{estimate['documents_exceeding_context']} document(s) dépasse(nt) "
                f"la fenêtre de contexte du modèle {model} ({estimate['context_window']:,} tokens)."
            )

    # Sources suspectes (anti-bot, contenu vide)
    suspects = [d for d in documents_info if d["Qualité"] == "Suspect"]
    if suspects:
        st.markdown("---")
        st.subheader("Sources suspectes")
        st.warning(
            f"{len(suspects)} source(s) détectée(s) comme suspecte(s) "
            "(contenu anti-bot possible ou texte trop court). "
            "Vérifiez le contenu ou supprimez les fichiers concernés."
        )
        for doc in suspects:
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"**{doc['Fichier']}** — Contenu suspect (anti-bot ou texte < 200 caractères)")
            with col_b:
                if st.button("Supprimer", key=f"del_suspect_{doc['Fichier']}"):
                    suspect_path = corpus_dir / doc["Fichier"]
                    if suspect_path.exists():
                        suspect_path.unlink()
                    st.rerun()

    # Déduplication
    if duplicates > 0:
        st.markdown("---")
        st.subheader("Gestion des doublons")
        for doc in documents_info:
            if doc["Doublon"] != "unique":
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.markdown(f"**{doc['Fichier']}** — {doc['Doublon']}")
                with col_b:
                    if st.button("Supprimer", key=f"del_{doc['Fichier']}"):
                        deduplicator.remove_duplicate(doc["Fichier"])
                        st.rerun()

    # Navigation inter-étapes
    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Retour à la configuration", use_container_width=True):
            st.session_state.current_page = "configuration"
            st.rerun()
    with col_next:
        if st.button("Continuer vers le plan →", type="primary", use_container_width=True):
            state = st.session_state.project_state
            if state:
                state.current_step = "plan"
                project_id = st.session_state.current_project
                if project_id:
                    save_json(PROJECTS_DIR / project_id / "state.json", state.to_dict())
            st.session_state.current_page = "plan"
            st.rerun()


def _render_metadata_overrides(corpus_dir: Path, project_dir: Path):
    """Phase 3: Metadata overrides and GROBID status per document."""
    config = st.session_state.project_state.config if st.session_state.get("project_state") else {}
    grobid_config = config.get("grobid", {})
    grobid_enabled = grobid_config.get("enabled", False)

    # Check if metadata store exists
    metadata_db = project_dir / "metadata.db"
    if not metadata_db.exists():
        return

    try:
        from src.core.metadata_store import MetadataStore
        store = MetadataStore(str(project_dir))
        all_docs = store.get_all_documents()
        store.close()
    except Exception:
        return

    if not all_docs:
        return

    with st.expander("Métadonnées des documents (Phase 3)", expanded=False):
        # GROBID status indicator
        if grobid_enabled:
            st.markdown("**GROBID** : Activé")
            server_url = grobid_config.get("server_url", "http://localhost:8070")
            try:
                import requests
                resp = requests.get(f"{server_url}/api/isalive", timeout=3)
                if resp.status_code == 200:
                    st.success(f"Serveur GROBID accessible ({server_url})")
                else:
                    st.warning(f"GROBID non accessible ({server_url})")
            except Exception:
                st.warning(f"Impossible de contacter GROBID ({server_url})")
        else:
            st.caption("GROBID désactivé — Activez-le dans Configuration > Phase 3")

        st.markdown("---")

        for doc in all_docs:
            doc_id = doc.doc_id if hasattr(doc, 'doc_id') else doc.get("doc_id", "")
            title = (doc.title if hasattr(doc, 'title') else doc.get("title")) or doc_id
            authors = (doc.authors if hasattr(doc, 'authors') else doc.get("authors")) or ""
            year = (doc.year if hasattr(doc, 'year') else doc.get("year")) or ""

            col_info, col_action = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{title}**")
                st.caption(f"Auteurs : {authors or 'Non renseigné'} | Année : {year or 'Non renseignée'}")
                if grobid_enabled:
                    grobid_status = (doc.grobid_status if hasattr(doc, 'grobid_status') else None) or "non traité"
                    status_icon = {"processed": "green", "failed": "red", "non traité": "gray"}.get(grobid_status, "gray")
                    st.markdown(f"GROBID : :{status_icon}[{grobid_status}]")

            with col_action:
                if st.button("Corriger", key=f"override_{doc_id}"):
                    st.session_state[f"_editing_meta_{doc_id}"] = True

            # Inline editing form
            if st.session_state.get(f"_editing_meta_{doc_id}", False):
                with st.form(key=f"meta_form_{doc_id}"):
                    new_title = st.text_input("Titre", value=title or "", key=f"meta_title_{doc_id}")
                    new_authors = st.text_input("Auteurs", value=authors or "", key=f"meta_authors_{doc_id}")
                    try:
                        year_val = int(year) if year else 2024
                    except (ValueError, TypeError):
                        year_val = 2024
                    new_year = st.number_input("Année", value=year_val, min_value=1900, max_value=2100, key=f"meta_year_{doc_id}")

                    col_submit, col_cancel = st.columns(2)
                    with col_submit:
                        submitted = st.form_submit_button("Sauvegarder", type="primary")
                    with col_cancel:
                        cancelled = st.form_submit_button("Annuler")

                    if submitted:
                        try:
                            from src.core.metadata_overrides import MetadataOverrides
                            overrides = MetadataOverrides(project_dir=project_dir)
                            overrides.set_override(
                                doc_id=doc_id,
                                title=new_title if new_title != title else None,
                                authors=new_authors if new_authors != authors else None,
                                year=int(new_year) if str(new_year) != str(year) else None,
                            )
                            st.success(f"Métadonnées corrigées pour {doc_id}")
                            st.session_state[f"_editing_meta_{doc_id}"] = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur : {e}")
                    if cancelled:
                        st.session_state[f"_editing_meta_{doc_id}"] = False
                        st.rerun()

            st.markdown("---")


def _display_report(report: AcquisitionReport):
    """Affiche le rapport d'acquisition."""
    if report.successful > 0:
        st.success(f"{report.successful} document(s) acquis avec succès.")
    if report.failed > 0:
        st.warning(f"{report.failed} document(s) en échec.")

    for status in report.statuses:
        if status.status == "SUCCESS":
            st.markdown(f"**{status.source}** — {status.message}")
        else:
            st.markdown(f"**{status.source}** — {status.message}")


def _render_rag_inspection(project_dir: Path):
    """Section d'inspection du corpus RAG (chunks indexés)."""
    chromadb_dir = project_dir / "chromadb"
    metadata_db_path = project_dir / "metadata.db"

    has_chromadb = chromadb_dir.exists() and any(chromadb_dir.iterdir()) if chromadb_dir.exists() else False
    has_metadata = metadata_db_path.exists()

    if not has_chromadb and not has_metadata:
        return

    with st.expander("Inspection du corpus RAG"):
        # Informations de stockage
        st.markdown("**Emplacements des données**")
        st.code(f"ChromaDB : {chromadb_dir}\nMetadata SQLite : {metadata_db_path}")

        col1, col2 = st.columns(2)

        # Nombre de chunks ChromaDB
        chroma_count = 0
        if has_chromadb:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(chromadb_dir))
                collection = client.get_or_create_collection("orchestria_corpus")
                chroma_count = collection.count()
            except Exception:
                pass
        col1.metric("Chunks ChromaDB", chroma_count)

        # Nombre de chunks SQLite
        sqlite_count = 0
        if has_metadata:
            try:
                from src.core.metadata_store import MetadataStore
                store = MetadataStore(str(project_dir))
                sqlite_count = store.count_chunks()
                store.close()
            except Exception:
                pass
        col2.metric("Chunks SQLite", sqlite_count)

        # Liste des chunks par document
        if has_metadata and sqlite_count > 0:
            st.markdown("**Chunks par document**")
            try:
                import pandas as pd
                from src.core.metadata_store import MetadataStore
                store = MetadataStore(str(project_dir))
                all_chunks = store.get_all_chunks()
                store.close()

                if all_chunks:
                    chunk_data = []
                    for c in all_chunks[:100]:  # Limiter à 100 pour la performance
                        chunk_data.append({
                            "Document": c.get("doc_id", ""),
                            "Chunk ID": c.get("chunk_id", ""),
                            "Texte (extrait)": (c.get("text", "")[:200] + "...") if len(c.get("text", "")) > 200 else c.get("text", ""),
                            "Page": c.get("page_number", ""),
                            "Tokens": c.get("token_count", ""),
                        })
                    df = pd.DataFrame(chunk_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    if len(all_chunks) > 100:
                        st.caption(f"Affichage limité aux 100 premiers chunks sur {len(all_chunks)} au total.")
            except Exception as e:
                st.warning(f"Impossible de charger les chunks : {e}")

        # Recherche test dans le RAG
        if has_chromadb and chroma_count > 0:
            st.markdown("**Recherche test**")
            test_query = st.text_input("Requête de test", placeholder="Saisissez un terme pour tester la recherche RAG...")
            if test_query and st.button("Tester la recherche"):
                try:
                    from src.core.rag_engine import RAGEngine
                    config = st.session_state.project_state.config if st.session_state.get("project_state") else {}
                    engine = RAGEngine(
                        persist_dir=chromadb_dir,
                        config=config,
                        top_k=config.get("rag", {}).get("top_k", 10),
                        relevance_threshold=config.get("rag", {}).get("relevance_threshold", 0.3),
                    )
                    result = engine.search(test_query, top_k=5)
                    if result.chunks:
                        for i, chunk in enumerate(result.chunks):
                            similarity = chunk.get("similarity", 0)
                            source = chunk.get("source_file", "inconnu")
                            text = chunk.get("text", "")
                            st.markdown(f"**Résultat {i+1}** (score: {similarity:.3f}) — {source}")
                            st.text(text[:300] + ("..." if len(text) > 300 else ""))
                    else:
                        st.info("Aucun résultat trouvé.")
                except Exception as e:
                    st.error(f"Erreur de recherche : {e}")
