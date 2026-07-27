"""
app.py
------
Streamlit prototype for the Disaster Relief Resource Matching Agent.

Run with:
    streamlit run app/app.py
"""

import sys
import os
import time
import random
import pandas as pd
import streamlit as st
import pydeck as pdk
from dotenv import load_dotenv

# allow importing from src/ when run as `streamlit run app/app.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load ANTHROPIC_API_KEY / GROQ_API_KEY / RELIEFWEB_APPNAME from a .env file
# in the project root, if one exists. Without this, os.environ.get(...) in
# ai_message.py / groq_triage.py never sees keys that only live in a .env
# file -- it only sees real OS environment variables. This must run before
# any src module reads os.environ.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from src.preprocessing import load_needs, load_resources
from src.urgency import compute_urgency_score, urgency_level, rank_needs_by_urgency, compute_model_confidence
from src.matching import match_need_with_resources
from src.guardrails import apply_guardrails, validate_need
from src.ai_message import generate_coordination_message
from src.memory import log_decision, get_recent_logs
from src.groq_triage import fast_triage
from src.rag import retrieve_context
from src.geo import get_coords, jitter_coords

st.set_page_config(page_title="Disaster Relief Resource Matching Agent", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

URGENCY_COLORS = {
    "HIGH": [220, 40, 40, 200],
    "MEDIUM": [255, 165, 0, 200],
    "LOW": [50, 180, 80, 200],
}


def build_deck(resource_points, need_points, arc_points):
    """Builds a pydeck Deck from plain-dict point/arc lists. Any of the
    three inputs may be empty -- the map still renders (e.g. resources only,
    before any simulation has run)."""
    layers = []
    if resource_points:
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=resource_points,
            get_position="[lon, lat]", get_radius="radius",
            get_fill_color=[70, 130, 220, 160], pickable=True,
        ))
    if need_points:
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=need_points,
            get_position="[lon, lat]", get_radius="radius",
            get_fill_color="color", pickable=True,
        ))
    if arc_points:
        layers.append(pdk.Layer(
            "ArcLayer", data=arc_points,
            get_source_position="[src_lon, src_lat]",
            get_target_position="[tgt_lon, tgt_lat]",
            get_source_color=[255, 200, 0, 220],
            get_target_color=[0, 200, 255, 220],
            get_width=5,
            width_min_pixels=3,
        ))
    view_state = pdk.ViewState(latitude=22.0, longitude=79.0, zoom=4.2, pitch=25)
    return pdk.Deck(layers=layers, initial_view_state=view_state,
                     tooltip={"html": "{tooltip}", "style": {"color": "white"}})


st.title("🆘 Disaster Relief Resource Matching Agent")
st.caption("Matches reported needs with available resources and prepares a coordination "
           "message for a human to review and approve. No dispatch happens automatically.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Submit a Need", "📊 Batch Run (All Needs)", "🧾 Agent Logs", "🗺️ Live Map & Simulation",
])

# ---------------------------------------------------------------------------
# TAB 1: Single need submission
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Report a Need")

    try:
        resources_df = load_resources(os.path.join(DATA_DIR, "resources.csv"))
    except Exception as e:
        st.error(f"Could not load resources.csv: {e}")
        resources_df = pd.DataFrame()

    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input("Location", value="Delhi")
        category = st.selectbox("Category", ["food", "medicine", "shelter", "water", "transport"])
        quantity = st.number_input("Quantity needed", min_value=1, value=50)
    with col2:
        urgency = st.slider("Reported urgency (0-10)", 0, 10, 7)
        description = st.text_area("Description (optional, helps AI scoring)",
                                    placeholder="e.g. Families trapped without food for 2 days")

    if st.button("🔎 Match Resources", type="primary"):
        need = {
            "id": "manual-entry",
            "location": location,
            "category": category,
            "quantity": quantity,
            "urgency": urgency,
            "description": description,
        }

        # Groq fast triage runs immediately, on every submission -- independent
        # of whether the need later passes validation/matching. This is the
        # "instant read" half of the dual-model architecture: low-latency
        # feedback while the rest of the (slower) pipeline still runs.
        with st.spinner("Fast triage..."):
            triage = fast_triage(need)
        triage_icon = "⚡" if triage["source"] == "groq" else "🔧"
        st.caption(
            f"{triage_icon} Groq fast triage: urgency_hint={triage['urgency_hint']}, "
            f"category_guess={triage['category_guess']} — {triage['reasoning']} "
            f"({'live Groq call' if triage['source'] == 'groq' else 'keyword fallback, no GROQ_API_KEY or call failed'})"
        )

        is_valid, error = validate_need(need)
        if not is_valid:
            st.error(f"Guardrail rejected input: {error}")
        else:
            need["urgency_score"] = compute_urgency_score(need)
            need["urgency_level"] = urgency_level(need["urgency_score"])

            resources = resources_df.to_dict(orient="records")
            candidates = match_need_with_resources(need, resources)
            best = candidates[0] if candidates else None

            match_result = {
                "need": need,
                "best_match": best[0] if best else None,
                "score": best[1] if best else 0.0,
            }
            checked = apply_guardrails(match_result)

            st.markdown(f"### Status: `{checked['status']}`")

            badge_color = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(need["urgency_level"], "⚪")
            st.write(f"**Computed urgency:** {badge_color} {need['urgency_level']} "
                     f"(score: {need['urgency_score']}/10)")

            confidence = compute_model_confidence(need)
            if confidence is not None:
                st.caption(f"🧠 Disaster-relevance classifier confidence: {confidence * 100:.1f}% "
                           f"(trained on Kaggle Disaster Tweets dataset)")
            elif description:
                st.caption("🧠 Classifier model not found — using keyword-based fallback for text boost. "
                           "Run `python -m src.train_classifier` to enable the trained model.")

            if checked["status"] == "MATCHED":
                st.success("Match found — awaiting human approval before dispatch.")
                st.json({"resource": checked["resource"], "confidence": checked["score"]})

                with st.spinner("Retrieving live + historical context (RAG)..."):
                    rag_context = retrieve_context(need)

                if rag_context["live_reports"]:
                    with st.expander(f"🌐 Live ReliefWeb context used ({len(rag_context['live_reports'])} reports)"):
                        for r in rag_context["live_reports"]:
                            st.write(f"- **{r['title']}** ({r.get('date', 'N/A')[:10]}) — {r['source']}")
                if rag_context["historical_examples"]:
                    with st.expander(f"📚 Similar historical examples used ({len(rag_context['historical_examples'])})"):
                        for ex in rag_context["historical_examples"]:
                            st.write(f"- \"{ex['text'][:200]}\" (similarity: {ex['similarity']})")

                with st.spinner("Generating coordination message..."):
                    message = generate_coordination_message(
                        need, checked["resource"], checked["score"],
                        rag_context=rag_context, groq_triage=triage,
                    )
                checked["ai_message"] = message
                st.info(f"**Coordination message:**\n\n{message}")

                c1, c2 = st.columns(2)
                if c1.button("✅ Approve Dispatch (human-in-loop)"):
                    st.success("Approved — this would trigger real dispatch/notification in production.")
                if c2.button("❌ Reject Match"):
                    st.warning("Match rejected by reviewer.")

            elif checked["status"] == "NO_MATCH":
                st.error(checked["reason"])
                if checked.get("escalate_to_authority"):
                    st.warning("⚠️ High urgency with no match — flagged for escalation to emergency authority.")
                checked["ai_message"] = "No match — escalation flagged." if checked.get("escalate_to_authority") else "No match found."

            else:
                st.warning(f"{checked['status']}: {checked['reason']}")
                checked["ai_message"] = checked["reason"]

            log_decision(checked, ai_message=checked.get("ai_message", ""))

# ---------------------------------------------------------------------------
# TAB 2: Batch run over needs.csv
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Run Agent Over All Needs in data/needs.csv")

    if st.button("▶️ Run Full Batch"):
        try:
            needs_df = load_needs(os.path.join(DATA_DIR, "needs.csv"))
            resources_df = load_resources(os.path.join(DATA_DIR, "resources.csv"))
        except Exception as e:
            st.error(f"Data loading error: {e}")
        else:
            ranked = rank_needs_by_urgency(needs_df)
            resources = resources_df.to_dict(orient="records")

            rows = []
            for need in ranked:
                candidates = match_need_with_resources(need, resources)
                best = candidates[0] if candidates else None
                match_result = {"need": need, "best_match": best[0] if best else None,
                                 "score": best[1] if best else 0.0}
                checked = apply_guardrails(match_result)
                log_decision(checked)

                rows.append({
                    "Location": need["location"],
                    "Category": need["category"],
                    "Qty Needed": need["quantity"],
                    "Urgency": need.get("urgency_level", "N/A"),
                    "Status": checked["status"],
                    "Matched Resource": checked.get("resource", {}).get("contact", "-")
                              if checked["status"] == "MATCHED" else "-",
                    "Confidence": checked.get("score", 0.0),
                })

            result_df = pd.DataFrame(rows)
            st.dataframe(result_df, use_container_width=True)

            matched = (result_df["Status"] == "MATCHED").sum()
            st.metric("Needs matched", f"{matched} / {len(result_df)}",
                       f"{round(100 * matched / len(result_df))}% coverage")

# ---------------------------------------------------------------------------
# TAB 3: Logs
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Recent Agent Decisions (SQLite log)")
    logs = get_recent_logs(limit=25)
    if logs:
        st.dataframe(pd.DataFrame(logs)[["timestamp", "need_id", "status", "score", "message"]],
                     use_container_width=True)
    else:
        st.info("No logs yet — submit a need or run a batch to populate this table.")

# ---------------------------------------------------------------------------
# TAB 4: Live map + simulated incoming disaster feed
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Live Map & Incoming Disaster Feed Simulation")
    st.caption(
        "Streams needs in one at a time, like a real intake system, and runs each through "
        "the full pipeline (Groq triage → matching → guardrails) live on the map. The single "
        "highest-urgency match at the end gets the full Groq → RAG → Claude treatment as a "
        "spotlight case, so the multi-model pipeline's actual output is visible, not just implied."
    )

    if "sim_events" not in st.session_state:
        st.session_state.sim_events = []

    try:
        resources_df2 = load_resources(os.path.join(DATA_DIR, "resources.csv"))
    except Exception as e:
        st.error(f"Could not load resources.csv: {e}")
        resources_df2 = pd.DataFrame()

    resource_points = []
    resource_coords_by_id = {}
    for r in resources_df2.to_dict(orient="records"):
        base = get_coords(r["location"])
        if base:
            lat, lon = jitter_coords(base[0], base[1], seed=f"resource-{r['id']}")
            resource_coords_by_id[r["id"]] = (lat, lon)
            resource_points.append({
                "lat": lat, "lon": lon, "radius": 12000,
                "tooltip": f"📦 {r['contact']} — {r['category']} x{r['quantity']} ({r['status']})",
            })

    col_a, col_b, _ = st.columns([1.4, 1, 2])
    run_sim = col_a.button("▶️ Simulate Incoming Disaster Feed", type="primary")
    reset_sim = col_b.button("🔄 Reset")

    if reset_sim:
        st.session_state.sim_events = []
        st.rerun()

    map_placeholder = st.empty()
    metrics_placeholder = st.empty()
    feed_placeholder = st.empty()
    spotlight_placeholder = st.empty()

    def render_sim_state():
        need_points, arc_points = [], []
        for ev in st.session_state.sim_events:
            base_coords = ev["coords"]
            if not base_coords:
                continue
            need_id = ev["need"].get("id", "manual-entry")
            coords = jitter_coords(base_coords[0], base_coords[1], seed=f"need-{need_id}")
            level = ev["need"].get("urgency_level", "LOW")
            need_points.append({
                "lat": coords[0], "lon": coords[1],
                "radius": 20000 + min(ev["need"].get("quantity", 10), 200) * 150,
                "color": URGENCY_COLORS.get(level, [120, 120, 120, 200]),
                "tooltip": (f"🆘 {ev['need']['category'].title()} needed in {ev['need']['location']} "
                            f"— {level} urgency — {ev['checked']['status']}"),
            })
            if ev["checked"]["status"] == "MATCHED":
                res_id = ev["checked"]["resource"].get("id")
                res_coords = resource_coords_by_id.get(res_id)
                if res_coords:
                    arc_points.append({
                        "src_lat": coords[0], "src_lon": coords[1],
                        "tgt_lat": res_coords[0], "tgt_lon": res_coords[1],
                    })

        map_placeholder.pydeck_chart(build_deck(resource_points, need_points, arc_points),
                                      use_container_width=True)

        total = len(st.session_state.sim_events)
        matched = sum(1 for e in st.session_state.sim_events if e["checked"]["status"] == "MATCHED")
        escalations = sum(1 for e in st.session_state.sim_events if e["checked"].get("escalate_to_authority"))
        with metrics_placeholder.container():
            m1, m2, m3 = st.columns(3)
            m1.metric("Needs processed", total)
            m2.metric("Matched", matched)
            m3.metric("Escalations flagged", escalations)

        with feed_placeholder.container():
            st.markdown("**Live feed** (most recent first)")
            if not st.session_state.sim_events:
                st.caption("No events yet — click Simulate to start streaming needs in.")
            status_icon = {"MATCHED": "✅", "NO_MATCH": "🚨",
                           "RESOURCE_UNAVAILABLE": "⚠️", "LOW_CONFIDENCE": "🟡"}
            urgency_badge = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}
            for ev in reversed(st.session_state.sim_events[-8:]):
                icon = status_icon.get(ev["checked"]["status"], "⬜")
                badge = urgency_badge.get(ev["need"].get("urgency_level"), "⚪")
                dest = (f" → **{ev['checked']['resource']['contact']}**"
                        if ev["checked"]["status"] == "MATCHED" else "")
                st.write(f"{icon} {badge} **{ev['need']['category'].title()}** in "
                         f"**{ev['need']['location']}** (qty {ev['need']['quantity']}) — "
                         f"{ev['checked']['status']}{dest}")

    render_sim_state()

    if run_sim:
        try:
            needs_df2 = load_needs(os.path.join(DATA_DIR, "needs.csv"))
        except Exception as e:
            st.error(f"Could not load needs.csv: {e}")
        else:
            st.session_state.sim_events = []
            resources_list = resources_df2.to_dict(orient="records")
            shuffled = needs_df2.to_dict(orient="records")
            random.shuffle(shuffled)

            for raw_need in shuffled:
                need = dict(raw_need)
                need["urgency_score"] = compute_urgency_score(need)
                need["urgency_level"] = urgency_level(need["urgency_score"])

                triage = fast_triage(need)

                candidates = match_need_with_resources(need, resources_list)
                best = candidates[0] if candidates else None
                match_result = {"need": need, "best_match": best[0] if best else None,
                                 "score": best[1] if best else 0.0}
                checked = apply_guardrails(match_result)
                log_decision(checked)

                st.session_state.sim_events.append({
                    "need": need, "checked": checked,
                    "coords": get_coords(need["location"]), "triage": triage,
                })

                render_sim_state()
                time.sleep(1.1)

            st.success(f"Simulation complete — {len(st.session_state.sim_events)} incoming needs processed.")

            # Spotlight: the ONE case that gets the full, slow, expensive
            # treatment -- Groq's triage read + live RAG retrieval + Claude's
            # final message -- displayed in full so the multi-model pipeline's
            # actual output is visible in one place, not scattered across
            # collapsed expanders.
            matched_events = [e for e in st.session_state.sim_events if e["checked"]["status"] == "MATCHED"]
            if matched_events:
                spotlight = max(matched_events, key=lambda e: e["need"]["urgency_score"])
                sn, sc = spotlight["need"], spotlight["checked"]
                with spotlight_placeholder.container():
                    st.markdown("---")
                    st.markdown("### 🔎 Spotlight case: highest-urgency match this run")
                    st.write(f"**{sn['category'].title()}** needed in **{sn['location']}** "
                             f"(urgency: {sn['urgency_level']}, qty {sn['quantity']})")
                    with st.spinner("Running full pipeline: Groq triage → RAG retrieval → Claude message..."):
                        rag_ctx = retrieve_context(sn)
                        message = generate_coordination_message(
                            sn, sc["resource"], sc["score"],
                            rag_context=rag_ctx, groq_triage=spotlight["triage"],
                        )
                    cap_bits = []
                    if rag_ctx["live_reports"]:
                        cap_bits.append(f"🌐 {len(rag_ctx['live_reports'])} live ReliefWeb report(s)")
                    if rag_ctx["historical_examples"]:
                        cap_bits.append(f"📚 {len(rag_ctx['historical_examples'])} historical example(s)")
                    if cap_bits:
                        st.caption("Grounded with: " + " · ".join(cap_bits))
                    else:
                        st.caption("No external context retrieved for this case — message generated from need + resource only.")
                    st.info(message)
            else:
                st.caption("No MATCHED needs this run to spotlight — try again, matches depend on shuffle order.")

st.markdown("---")
st.caption("⚠️ Limitations: synthetic sample data by default; urgency boosting uses a simple "
           "keyword heuristic, not a trained classifier; all matches require human approval "
           "before any real-world dispatch.")
