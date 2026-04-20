"""
Built-in benchmark datasets for RAG evaluation.
Covers different domains and difficulty levels.
"""
from core.models import RAGSample
from typing import List


def get_science_qa_dataset() -> List[RAGSample]:
    """Science Q&A dataset with high-quality ground truth."""
    return [
        RAGSample(
            sample_id="sci_001",
            question="What is photosynthesis and what are its main inputs and outputs?",
            ground_truth="Photosynthesis is the process by which plants convert light energy into chemical energy. The main inputs are carbon dioxide (CO2), water (H2O), and sunlight. The outputs are glucose (C6H12O6) and oxygen (O2). The overall equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.",
            contexts=[
                "Photosynthesis is a biological process used by plants, algae, and cyanobacteria to convert light energy, usually from the sun, into chemical energy stored in glucose. This process occurs primarily in the chloroplasts of plant cells.",
                "The inputs required for photosynthesis are: (1) Carbon dioxide (CO2) absorbed through tiny pores called stomata, (2) Water (H2O) absorbed through roots, and (3) Light energy captured by chlorophyll pigments. The outputs produced are glucose, which stores energy for the plant, and oxygen as a byproduct.",
                "The chemical equation for photosynthesis: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. This means six molecules of carbon dioxide and six molecules of water are converted into one molecule of glucose and six molecules of oxygen."
            ],
            metadata={"domain": "biology", "difficulty": "medium"}
        ),
        RAGSample(
            sample_id="sci_002",
            question="How does the speed of light relate to Einstein's theory of special relativity?",
            ground_truth="Einstein's special relativity is built on two postulates: the laws of physics are the same in all inertial frames, and the speed of light in a vacuum (approximately 3×10^8 m/s) is constant for all observers regardless of their motion or the motion of the source. This leads to time dilation and length contraction at high speeds.",
            contexts=[
                "Albert Einstein's special theory of relativity, published in 1905, revolutionized our understanding of space and time. The theory rests on two fundamental postulates: first, the laws of physics are identical in all inertial (non-accelerating) reference frames; second, the speed of light in a vacuum is always approximately 299,792,458 meters per second (about 3×10^8 m/s), regardless of the motion of the light source or the observer.",
                "A key consequence of special relativity is time dilation: moving clocks tick slower relative to stationary observers. Another consequence is length contraction: objects moving at high speeds appear shorter in the direction of motion. These effects become significant only at speeds approaching the speed of light.",
                "The famous equation E=mc² also derives from special relativity, showing that energy (E) equals mass (m) times the speed of light (c) squared. This demonstrates the equivalence of mass and energy."
            ],
            metadata={"domain": "physics", "difficulty": "hard"}
        ),
        RAGSample(
            sample_id="sci_003",
            question="What causes earthquakes?",
            ground_truth="Earthquakes are caused by the movement of tectonic plates. When stress accumulated along fault lines is suddenly released, it generates seismic waves. Most earthquakes occur at tectonic plate boundaries. The point of origin underground is called the hypocenter, and the point on the surface directly above is the epicenter.",
            contexts=[
                "Earthquakes occur when rocks in Earth's crust fracture and slip along faults due to tectonic forces. The Earth's lithosphere is divided into several large tectonic plates that are constantly moving, driven by heat from Earth's interior. Most seismic activity occurs where these plates interact.",
                "The three main types of plate boundaries where earthquakes commonly occur are: convergent boundaries (where plates collide), divergent boundaries (where plates move apart), and transform boundaries (where plates slide past each other horizontally). The San Andreas Fault in California is a famous transform boundary.",
                "When an earthquake occurs, the point of initial rupture underground is called the hypocenter or focus. Directly above on the surface is the epicenter. Seismic energy radiates outward from the hypocenter as different types of waves: P-waves (primary/compressional) and S-waves (secondary/shear waves)."
            ],
            metadata={"domain": "geology", "difficulty": "easy"}
        ),
    ]


def get_tech_qa_dataset() -> List[RAGSample]:
    """Technology Q&A dataset."""
    return [
        RAGSample(
            sample_id="tech_001",
            question="What is a transformer model in machine learning?",
            ground_truth="A transformer is a deep learning architecture introduced in 2017 that uses self-attention mechanisms to process sequential data. Unlike RNNs, transformers process all tokens in parallel, making them more efficient to train. They consist of encoder and decoder stacks with multi-head attention layers and feed-forward networks. Transformers are the basis for modern language models like BERT and GPT.",
            contexts=[
                "The Transformer model was introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al. It revolutionized natural language processing by replacing recurrent neural networks (RNNs) with a self-attention mechanism that can process entire sequences simultaneously rather than sequentially.",
                "The key innovation of transformers is multi-head self-attention, which allows the model to attend to different positions of the input sequence simultaneously. A transformer consists of an encoder (processes input) and a decoder (generates output), each built from stacked layers of multi-head attention and position-wise feed-forward networks.",
                "Transformers have become the dominant architecture in NLP. BERT (Bidirectional Encoder Representations from Transformers) uses only the encoder for tasks like classification. GPT (Generative Pre-trained Transformer) uses only the decoder for text generation. These models are pre-trained on massive text corpora and fine-tuned for specific tasks."
            ],
            metadata={"domain": "ml", "difficulty": "medium"}
        ),
        RAGSample(
            sample_id="tech_002",
            question="How does HTTPS encryption work?",
            ground_truth="HTTPS uses TLS (Transport Layer Security) to encrypt web traffic. During the TLS handshake, the server presents a certificate containing its public key. The client verifies the certificate, generates a session key, encrypts it with the server's public key, and sends it over. Both parties then use this shared session key for symmetric encryption of all subsequent data.",
            contexts=[
                "HTTPS (HyperText Transfer Protocol Secure) is HTTP with TLS/SSL encryption layered on top. The TLS protocol provides three main security guarantees: encryption (data is unreadable to interceptors), authentication (you're talking to the real server, not an impersonator), and integrity (data hasn't been tampered with in transit).",
                "The TLS handshake process: (1) Client sends 'hello' with supported cipher suites. (2) Server responds with its digital certificate containing its public key. (3) Client verifies the certificate with a trusted Certificate Authority (CA). (4) Client generates a pre-master secret, encrypts it with the server's public key, and sends it. (5) Both derive the same session keys from the pre-master secret. (6) All subsequent communication is encrypted with fast symmetric encryption.",
                "Modern TLS (version 1.3) uses forward secrecy, meaning each session uses a unique ephemeral key pair. Even if a server's private key is later compromised, past sessions cannot be decrypted. The most common symmetric cipher used is AES-256-GCM."
            ],
            metadata={"domain": "security", "difficulty": "hard"}
        ),
        RAGSample(
            sample_id="tech_003",
            question="What is the difference between SQL and NoSQL databases?",
            ground_truth="SQL databases are relational, use structured schemas with tables, rows, and columns, and support ACID transactions. They're best for complex queries and structured data. NoSQL databases are non-relational and come in four types: document, key-value, column-family, and graph. They offer flexible schemas, horizontal scalability, and are better for unstructured or rapidly changing data at scale.",
            contexts=[
                "SQL (relational) databases organize data into structured tables with predefined schemas. They use SQL (Structured Query Language) for querying and support ACID properties: Atomicity, Consistency, Isolation, Durability. Examples include PostgreSQL, MySQL, and SQLite. They excel at complex queries with JOINs across multiple tables.",
                "NoSQL databases emerged to handle use cases that relational databases struggled with: massive scale, unstructured data, and rapid schema changes. The four main types are: (1) Document stores (MongoDB, CouchDB) - store data as JSON-like documents; (2) Key-value stores (Redis, DynamoDB) - simple key-value pairs; (3) Column-family stores (Cassandra, HBase) - optimized for column-oriented queries; (4) Graph databases (Neo4j) - store nodes and relationships.",
                "Choosing between SQL and NoSQL depends on your use case. SQL is better when you need complex queries, transactions across multiple records, and data integrity. NoSQL is better when you need horizontal scalability to handle massive write loads, flexible schemas that can evolve quickly, or specific data models like graphs or time-series."
            ],
            metadata={"domain": "databases", "difficulty": "medium"}
        ),
    ]


def get_hallucination_test_dataset() -> List[RAGSample]:
    """
    Dataset specifically designed to test hallucination detection.
    Includes cases where good pipelines should NOT hallucinate.
    """
    return [
        RAGSample(
            sample_id="hall_001",
            question="What is the population of Mars?",
            ground_truth="Mars has no permanent human population. As of now, no humans have ever traveled to Mars. Only robotic spacecraft and rovers have visited the planet.",
            contexts=[
                "Mars is the fourth planet from the Sun in our solar system. It has a thin atmosphere composed primarily of carbon dioxide. As of 2024, no humans have visited Mars. Several robotic missions have successfully landed on Mars, including NASA's Perseverance rover.",
                "Current Mars exploration involves only uncrewed spacecraft. The Mars 2020 mission deployed the Perseverance rover to study the planet's habitability. Various space agencies have plans for future crewed Mars missions, but none have been executed."
            ],
            metadata={"domain": "astronomy", "difficulty": "easy", "test_type": "hallucination_resistance"}
        ),
        RAGSample(
            sample_id="hall_002",
            question="Who invented the telephone?",
            ground_truth="Alexander Graham Bell is widely credited with inventing the telephone and received the first patent for it in 1876. However, this is historically disputed — Italian inventor Antonio Meucci filed a caveat for a voice communication device in 1871, and Elisha Gray filed a patent caveat the same day as Bell.",
            contexts=[
                "Alexander Graham Bell received US Patent 174,465 for the telephone on March 7, 1876. The patent was titled 'The method of, and apparatus for, transmitting vocal or other sounds telegraphically.' Bell's first successful voice transmission was on March 10, 1876.",
                "The invention of the telephone is historically contested. Antonio Meucci, an Italian inventor, developed a voice communication device and filed a caveat with the US Patent Office in 1871, but could not afford to renew it. In 2002, the US Congress passed a resolution recognizing Meucci's contributions to the invention of the telephone.",
                "Elisha Gray, co-founder of Western Electric, also filed a patent caveat for a telephone design on February 14, 1876 — the same day Bell filed his patent application. The priority dispute between Gray and Bell led to lengthy legal battles."
            ],
            metadata={"domain": "history", "difficulty": "medium", "test_type": "nuance"}
        ),
    ]


def get_all_datasets() -> List[RAGSample]:
    """Return all built-in datasets combined."""
    return (
        get_science_qa_dataset() +
        get_tech_qa_dataset() +
        get_hallucination_test_dataset()
    )


def get_corpus_documents() -> List[str]:
    """Return a corpus of documents for pipeline retrieval testing."""
    return [
        # Science
        "Photosynthesis is a biological process used by plants, algae, and cyanobacteria to convert light energy into chemical energy stored in glucose. Inputs: CO2, water, sunlight. Outputs: glucose, oxygen. Equation: 6CO2 + 6H2O + light → C6H12O6 + 6O2.",
        "Photosynthesis occurs in chloroplasts. The process has two stages: light-dependent reactions (in thylakoid membranes) and the Calvin cycle (in the stroma). Chlorophyll absorbs red and blue light, reflecting green light.",
        "Einstein's special relativity (1905) has two postulates: laws of physics are the same in all inertial frames, and the speed of light (3×10^8 m/s) is constant for all observers. Consequences include time dilation, length contraction, and E=mc².",
        "Earthquakes are caused by tectonic plate movements. Stress builds along fault lines and is suddenly released, generating seismic waves. The origin point underground is the hypocenter; the surface point above is the epicenter.",
        "Tectonic plate boundaries: convergent (plates collide, causing earthquakes and mountains), divergent (plates move apart, creating rift valleys), and transform (plates slide past each other, like the San Andreas Fault).",
        # Technology
        "The Transformer model (2017, 'Attention Is All You Need') uses self-attention to process sequences in parallel. Key components: multi-head attention, positional encoding, encoder-decoder architecture. Forms the basis of BERT and GPT.",
        "BERT uses only the transformer encoder for bidirectional context understanding. GPT uses only the decoder for autoregressive text generation. Both are pre-trained on large corpora and fine-tuned for downstream tasks.",
        "HTTPS uses TLS encryption. TLS handshake: server presents certificate with public key, client verifies with CA, client sends session key encrypted with server's public key, symmetric encryption used for data transfer.",
        "TLS 1.3 supports forward secrecy using ephemeral key pairs. Each session has a unique key, so compromising future keys doesn't expose past sessions. Uses AES-256-GCM symmetric cipher.",
        "SQL databases: relational, ACID transactions, structured schemas, complex JOINs. NoSQL types: document (MongoDB), key-value (Redis), column-family (Cassandra), graph (Neo4j). NoSQL offers horizontal scalability and flexible schemas.",
        # Astronomy
        "Mars is the fourth planet from the Sun. It has a thin atmosphere of CO2, two moons (Phobos and Deimos), and surface features including Olympus Mons (largest volcano in solar system) and Valles Marineris.",
        "No humans have visited Mars. Robotic missions include NASA's Perseverance rover (Mars 2020 mission), Curiosity rover, and various landers and orbiters. Mars has no permanent human population.",
        # History
        "Alexander Graham Bell received US Patent 174,465 for the telephone on March 7, 1876. First successful call was March 10, 1876. The invention is disputed — Antonio Meucci filed a caveat in 1871, and Elisha Gray filed on the same day as Bell.",
    ]
