# huffman-compression
Huffman Compression

A Python implementation of Huffman coding, a classic algorithm for lossless data compression. This project demonstrates how to compress and decompress files using Huffman trees, while also analyzing performance metrics like compression ratio and runtime.

📂 Project Structure
  - compress.py – Core compression & decompression logic:
  - Builds frequency dictionaries
  - Constructs Huffman trees
  - Encodes/decodes byte streams
  - Provides functions to compress/decompress files

🚀 Features
  - Build Huffman trees from frequency dictionaries
  - Generate prefix-free Huffman codes for symbols
  - Compress text and binary files into .huf format
  - Decompress .huf files back to original
  - Report compression ratio, runtime, and correctness checks
  - Supports arbitrary binary files (e.g., .bmp images tested in this project)

💡 Key Learnings
  - Implemented tree-based algorithms in Python
  - Worked with binary data and bit-level operations
  - Practiced algorithm analysis with time complexity and compression ratio
  - Reinforced skills in data structures, recursion, and file I/O

### 🛠️Getting Started
Clone the repo and navigate into the directory:
```
git clone https://github.com/<your-username>/huffman-compression.git
cd huffman-compression
```

Run compression or decompression interactively:
```
python compress.py
```
- Press c to compress → outputs filename.huf
- Press d to decompress → outputs filename.huf.orig

### 📊 Example Output
```
Processing: files/parrot.bmp
Compression Time: 0.0542 seconds
Decompression Time: 0.0391 seconds
Original Size: 307,254 bytes
Compressed Size: 198,745 bytes
Decompressed Size: 307,254 bytes
Compression Ratio: 64.72%
✅ SUCCESS: parrot.bmp decompressed correctly.
```
