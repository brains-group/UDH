from typing import List, Tuple
from pydantic import BaseModel, Field
from nkg.models.index_objects import EntityFingerprint
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy


    




class ExtractChunkEntities(dspy.Signature):
    __doc__ = """
    Extract every named entity from source_text as a deduplicated fingerprint list.
    This is the canonical entity pool for the chunk — all facts draw their entities from it.

    For each entity:
    - name: Canonical name. Resolve pronouns to proper referents (e.g., "you" → "Policyholder").
    - type: One of: PERSON, ORGANIZATION, LOCATION, DATE, POLICY, COVERAGE, DOCUMENT, CONCEPT.
    - role: 3-8 word micro-role describing its function in this specific text.
    - relational_anchors: Specific identifiers tied to this entity (policy numbers, codes, IDs). Empty list if none.
    """

    source_text: str = dspy.InputField()
    entities: list[EntityFingerprint] = dspy.OutputField(
        desc="Complete list of all entity fingerprints found in this chunk"
    )


class ShortEntityItem(BaseModel):
    name: str = Field(..., description="Canonical entity name. Resolve pronouns to proper referents.")
    role: str = Field(..., description="3-8 word micro-role describing this entity's function in the text.")


class ExtractChunkEntitiesShort(dspy.Signature):
    __doc__ = """
    Extract every named entity from source_text.
    For each entity provide:
    - name: Canonical name. Resolve pronouns to proper referents (e.g., "you" → "Policyholder").
    - role: 3-8 word micro-role describing its function in this specific text.
    Do not include type or relational anchors.
    """

    source_text: str = dspy.InputField()
    entities: list[ShortEntityItem] = dspy.OutputField(
        desc="Complete list of all entities found in this chunk"
    )


def extract_chunk_entities(text: str, use_cot: bool = True, full_profile: bool = True,
                            chunk_summary: str = "") -> list[EntityFingerprint]:
    """Extract entity fingerprints from a chunk. full_profile=False returns name+role only."""
    predict = dspy.ChainOfThought if use_cot else dspy.Predict
    if full_profile:
        return predict(ExtractChunkEntities)(source_text=text).entities
    else:
        short_items = predict(ExtractChunkEntitiesShort)(source_text=text).entities
        return [
            EntityFingerprint(
                name=item.name,
                type="",
                role=item.role,
                relational_anchors=(chunk_summary,) if chunk_summary else ()
            )
            for item in short_items
        ]
