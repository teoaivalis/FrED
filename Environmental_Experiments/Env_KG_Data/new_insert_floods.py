import os
import re
from neo4j import GraphDatabase

# === Neo4j Connection ===
URI = "neo4j://localhost:7689"
USERNAME = "neo4j"
PASSWORD = "br7ntoz426"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# === Predicate sets ===
TEXT_PREDICATES = {
    "hasCountry", "hasISO3", "hasDisasterType", "hasStatus", "hasDate",
    "hasLatitude", "hasLongitude", "hasGlide", "peopleAffected", "displacedPeople",
    "fatalitiesReported", "injuriesReported", "housesFlooded", "housesCollapsed",
    "schoolsAffected", "businessesDamaged", "roadsDamaged", "infrastructureDamaged",
    "economicDamage", "powerOutage", "waterSupplyCut", "agriculturalLandAffected",
    "needsShelter", "humanitarianNeeds", "causedBy", "weatherCause", "affectedArea",
    "hasWaters", "hasWetlands", "hasWoods", "hasScrubs", "hasGrasslands",
    "hasSprings", "hasTrees", "hasMountainRanges"
}

IMAGE_PREDICATES = {
    "hasvisualimage", "hasnirimage", "hasndwiimage",
    "hasLandCover", "showsFeature", "hasVegetationDensity",
    "capturedInSeason", "hasVegetationHealth", "hasVegetationMoisture",
    "hasMoistureLevel", "showsSignsOf", "includes", "traversedBy",
    "containsFeature", "isPartOf", "adjacentTo", "mayIndicate", "likelyRepresents",
    "experiencing",
    "imageType", "cloudCoverage"  # include image detail predicates too
}

# === Insert Logic ===
def insert_triple(tx, subject, predicate, object_):
    # Decide what the subject is:
    if subject.endswith("-visual") or subject.endswith("-nir") or subject.endswith("-ndwi"):
        # SUBJECT is an Image -> link it to an Attribute (its characteristic)
        query = f"""
        MERGE (i:Image {{name: $subject}})
        MERGE (a:Attribute {{name: $object}})
        MERGE (i)-[:{predicate}]->(a)
        """
    elif predicate == "hasCountry":
        query = """
        MERGE (d:Disaster {name: $subject})
        MERGE (r:Region {name: $object})
        MERGE (d)-[:hasCountry]->(r)
        """
    elif predicate in {"hasvisualimage", "hasnirimage", "hasndwiimage"}:
        # Disaster -> Image
        query = f"""
        MERGE (d:Disaster {{name: $subject}})
        MERGE (i:Image {{name: $object}})
        MERGE (d)-[:{predicate}]->(i)
        """
    else:
        # Default: Disaster -> Attribute
        query = f"""
        MERGE (d:Disaster {{name: $subject}})
        MERGE (a:Attribute {{name: $object}})
        MERGE (d)-[:{predicate}]->(a)
        """
    tx.run(query, subject=subject, object=object_)

# === File Processor ===
def process_file(file_path, mode):
    with driver.session() as session:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                clean_line = re.sub(r"[()]", "", line)
                if not clean_line.strip() or "," not in clean_line:
                    continue

                parts = [part.strip() for part in clean_line.strip().split(",")]
                if len(parts) != 3:
                    continue

                subject, predicate, object_ = parts

                # Validate predicate
                if mode == "text" and predicate not in TEXT_PREDICATES:
                    print(f"Skipping invalid text predicate: {predicate}")
                    continue
                if mode == "image" and predicate not in IMAGE_PREDICATES:
                    print(f"Skipping invalid image predicate: {predicate}")
                    continue

                # For images: enforce subject starts with 2 capital letters + '-' if NOT an image characteristic
                is_image_subject = subject.endswith("-visual") or subject.endswith("-nir") or subject.endswith("-ndwi")
                if mode == "image" and not is_image_subject and not re.match(r"^[A-Z]{2}-", subject):
                    print(f"Skipping image triple: subject must match XX- : {subject}")
                    continue

                # Handle possible lists in object
                object_list = [object_]
                for token in [" and ", " or ", ","]:
                    if token in object_:
                        object_list = [o.strip() for o in object_.split(token) if o.strip()]
                        break

                for obj in object_list:
                    session.execute_write(insert_triple, subject, predicate, obj)

# === Main ===
if __name__ == "__main__":
    print("=== Starting full triple insertion ===")

    # Text triples
    text_dir = "/home/theo_ai/Documents/extreme_events/neo4j_disasters/cleaned_new_disaster_full_triples"
    text_files = [os.path.join(text_dir, f) for f in os.listdir(text_dir) if f.endswith(".txt")]
    for file_path in text_files:
        print(f"Processing TEXT file: {file_path}")
        process_file(file_path, mode="text")

    # Visual triples
    #visual_dir = "after_collected_visual_triples"
    #visual_files = [os.path.join(visual_dir, f) for f in os.listdir(visual_dir) if f.endswith(".txt")]
    #for file_path in visual_files:
    #    print(f"Processing VISUAL IMAGE file: {file_path}")
    #    process_file(file_path, mode="image")

    # NIR triples
    #nir_dir = "after_collected_nir_triples"
    #nir_files = [os.path.join(nir_dir, f) for f in os.listdir(nir_dir) if f.endswith(".txt")]
    #for file_path in nir_files:
    #    print(f"Processing NIR IMAGE file: {file_path}")
    #    process_file(file_path, mode="image")

    # NDWI triples
    #ndwi_dir = "after_collected_ndwi_triples"
    #ndwi_files = [os.path.join(ndwi_dir, f) for f in os.listdir(ndwi_dir) if f.endswith(".txt")]
    #for file_path in ndwi_files:
    #    print(f"Processing NDWI IMAGE file: {file_path}")
    #    process_file(file_path, mode="image")

    driver.close()
    print("=== All triples inserted successfully ===")
