# %%
import os

import pandas as pd
import numpy as np
import fnmatch

# %%
TRANSCRIPTS_PATH = "/Users/kylie/Harvard University/Reach Every Reader - Integration for Scale - Converted Mursion Transcripts/"


def extract_what_did_you_hear(transcript: pd.DataFrame, response_number: int = 0):

    # Sometimes the speaker is missing, if so add blank speakers
    speaker_and_text = [
        ":" + line if ":" not in line else line for line in transcript.transcript
    ]

    # Split speaker from text on :
    speaker_and_text_lists = [line.split(":") for line in speaker_and_text]
    speakers = [item[0] for item in speaker_and_text_lists]
    text = [item[1] for item in speaker_and_text_lists]

    transcript["speaker"] = speakers
    transcript["text"] = text

    # if missing speaker add last speaker
    last_speaker = ""
    for line in transcript.index:
        if transcript.loc[line, "speaker"] == "":
            transcript.loc[line, "speaker"] = last_speaker
        last_speaker = transcript.loc[line, "speaker"]

    transcript["turn_of_talk"] = 0
    turn = 1
    last_speaker = ""
    for line in transcript.index:
        if transcript.loc[line, "speaker"] == last_speaker:
            transcript.loc[line, "turn_of_talk"] = turn
        elif transcript.loc[line, "speaker"] != last_speaker:
            turn = turn + 1
            transcript.loc[line, "turn_of_talk"] = turn
        last_speaker = transcript.loc[line, "speaker"]

    transcript["turn_text"] = transcript.groupby(["turn_of_talk"])["text"].transform(
        lambda x: " ".join(x)
    )
    turn_df = transcript.drop(axis="columns", labels="text")
    turn_df = turn_df.drop_duplicates(subset="turn_text")

    hear_df = turn_df[
        (
            turn_df.turn_text.str.contains("you hear ")
            & turn_df.speaker.str.contains("Reach Every Reader")
        )
        | (
            turn_df.turn_text.str.contains("you hear.")
            & turn_df.speaker.str.contains("Reach Every Reader")
        )
        | (
            turn_df.turn_text.str.contains("you hear?")
            & turn_df.speaker.str.contains("Reach Every Reader")
        )
        | (
            turn_df.turn_text.str.contains("you hear-")
            & turn_df.speaker.str.contains("Reach Every Reader")
        )
    ]

    hear_turns = list(hear_df.turn_of_talk)

    response_turns = [turn + 1 for turn in hear_turns]

    response_df = turn_df[turn_df.turn_of_talk.isin(response_turns)]
    response_df["response"] = np.arange(len(response_df))
    response_df = response_df.set_index("response")

    return response_df.loc[response_number, "turn_text"]


transcript = pd.read_excel(
    TRANSCRIPTS_PATH + "S78_EAD_1.25.21_A_transcript-converted.xlsx",
    header=None,
    names=["line", "time_stamp", "transcript"],
)

response = extract_what_did_you_hear(transcript=transcript, response_number=0)
# %%

doc_dicts = []
for filename in os.listdir(TRANSCRIPTS_PATH):
    if fnmatch.fnmatch(filename, "*EAD*-converted.xlsx") and not filename.startswith(
        "~$"
    ):
        try:
            transcript = pd.read_excel(
                TRANSCRIPTS_PATH + filename,
                header=None,
                names=["line", "time_stamp", "transcript"],
            )
            doc_dicts.append(
                {
                    "filename": filename,
                    "response": extract_what_did_you_hear(
                        transcript=transcript, response_number=0
                    ),
                }
            )
        except:
            print("Could not extract " + filename)

responses = pd.DataFrame(doc_dicts)
responses.to_csv("/Users/kylie/Dropbox/convert_transcripts/responses.csv")
# %%
