def create_diction_html(
    word_count,
    markers,
    honorific_display,
    verbal_style
):

    return f"""
    <div style='
        margin-bottom:16px;
        background:#eff6ff;
        border-left:5px solid #0284c7;
        padding:12px;
        border-radius:8px;
    '>

        <div style='
            font-weight:bold;
            color:#0369a1;
            margin-bottom:6px;
        '>

            Language and Speaking Patterns

            <span
                title='This section analyzes the words used in the spoken sentence, including regional words, respectful language, and the overall speaking style.'
                style='cursor:help; font-size:16px;'
            >
                ⓘ
            </span>

        </div>

        <div style='font-size:14px; color:#334155;'>

            • Words count:
            <strong>{word_count}</strong>

            <br>

            • Unique dialect words:
            <strong>{len(markers)}</strong>

            <br>

            • Honorifics:
            <strong>{honorific_display}</strong>

            <br>

            • Speaking style:
            <strong>{verbal_style}</strong>

        </div>

    </div>
    """


def create_prosody_html(
    particle_display,
    intonation_input,
    speech_rate
):

    # Make intonation easier to understand
    display_intonation = (
        "Unable to Determine"
        if intonation_input in [
            "Unknown",
            "Not Available",
            "Undetermined"
        ]
        else intonation_input
    )

    # Format speech rate for display
    display_speech_rate = (
        "Not Available"
        if speech_rate <= 0
        else f"{speech_rate:.2f} units/sec"
    )

    return f"""
    <div style='
        margin-bottom:16px;
        background:#fefce8;
        border-left:5px solid #ca8a04;
        padding:12px;
        border-radius:8px;
    '>

        <div style='
            font-weight:bold;
            color:#a16207;
            margin-bottom:6px;
        '>

            Voice and Speaking Pattern

            <span
                title='This section analyzes characteristics of the speaker voice, including sentence ending words, voice pattern, and speaking speed.'
                style='cursor:help; font-size:16px;'
            >
                ⓘ
            </span>

        </div>

        <div style='font-size:14px; color:#334155;'>

            • Sentence Ending Words

            <span
                title='These are words that commonly appear at the end of a sentence. They can add emphasis, tone, or additional meaning.'
                style='cursor:help; font-size:14px;'
            >
                ⓘ
            </span>:

            <strong>{particle_display}</strong>

            <br>

            • Intonation pattern:

            <strong>{display_intonation}</strong>

            <br>

            • Speaking Speed:

            <strong>{display_speech_rate}</strong>

        </div>

    </div>
    """


def create_ranking_html(sorted_scores):

    ranking_html = ""

    for index, (prov, score) in enumerate(sorted_scores):

        if index == 0:

            ranking_html += f"""
            <div style='
                margin-bottom:10px;
                padding:14px;
                background:#dcfce7;
                border-left:6px solid #16a34a;
                border-radius:8px;
            '>

                <div style='
                    font-size:16px;
                    font-weight:bold;
                    color:#166534;
                    margin-bottom:5px;
                '>
                    MOST SIMILAR DIALECT
                </div>

                <div style='
                    font-size:20px;
                    font-weight:bold;
                    color:#14532d;
                '>
                    {prov.capitalize()} — {score:.2f}%
                </div>

            </div>
            """

        else:

            ranking_html += f"""
            <div style='
                margin-bottom:6px;
                padding:5px;
                background:#f8fafc;
                border-left:4px solid #cbd5e1;
                border-radius:6px;
            '>

                {prov.capitalize()}:
                <strong>{score:.2f}%</strong>

            </div>
            """

    return ranking_html


def create_sbert_html(
    most_similar_province,
    most_similar_score,
    ranking_html
):

    return f"""
    <div style='
        margin-bottom:18px;
        background:#f8fafc;
        border-left:6px solid #6366f1;
        padding:14px;
        border-radius:8px;
    '>

        <div style='
            font-size:18px;
            font-weight:bold;
            color:#4338ca;
            margin-bottom:10px;
        '>

            Dialect Similarity

            <span
                title='The system compares the meaning of the spoken sentence with reference sentences from Batangas, Cavite, Laguna, Rizal, and Quezon using SBERT. A higher score means greater similarity to the available reference examples. This does not confirm the speaker actual province or dialect.'
                style='cursor:help; font-size:16px;'
            >
                ⓘ
            </span>

        </div>

        <div style='font-size:14px; color:#334155;'>

            • Closest Dialect Pattern:
            <strong>
                {most_similar_province.capitalize()}
            </strong>

            <br><br>

            • Similarity Score

            <span
                title='The similarity score shows how closely the spoken sentence matches the available reference sentences. A higher score means greater similarity.'
                style='cursor:help; font-size:14px;'
            >
                ⓘ
            </span>:

            <strong>
                {most_similar_score:.2f}%
            </strong>

        </div>

        <hr style="margin:12px 0">

        <div style='
            font-weight:bold;
            margin-bottom:10px;
            color:#334155;
        '>
            Dialect Similarity Ranking
        </div>

        {ranking_html}

    </div>
    """


def create_ann_html(
    accent_input,
    emotion_input
):

    return f"""
    <div style='
        margin-bottom:18px;
        background:#fff7ed;
        border-left:6px solid #ea580c;
        padding:14px;
        border-radius:8px;
    '>

        <div style='
            font-size:18px;
            font-weight:bold;
            color:#c2410c;
            margin-bottom:10px;
        '>

            ANN model Analysis

            <span
                title='This section uses Artificial Neural Network models to analyze audio characteristics and provide predictions for accent and emotion.'
                style='cursor:help; font-size:16px;'
            >
                ⓘ
            </span>

        </div>

        <div style='font-size:14px; color:#334155;'>

            • Detected Accent:
            <strong>{accent_input}</strong>

            <br><br>

            • Detected Emotion:
            <strong>{emotion_input}</strong>

        </div>

    </div>
    """