"""
This file reads binary SysEx data and converts it to a ML-friendly dataset.
It comprises a transpile of selected code from the `learnfm` project:

    https://github.com/bwhitman/learnfm/

and was entirely generated by ChatGPT 4.0, with guiding prompts from the author.

For an outline of the DX7 SysEx patch structure, see this useful note:

    https://homepages.abdn.ac.uk/d.j.benson/pages/dx7/sysex-format.txt
"""

import random

def unpack_patch(bulk):
    def read_byte(offset, mask=0b01111111, shift=0):
        return (bulk[offset] & mask) >> shift
    
    def read_char(offset):
        return chr(bulk[offset])
    
    operators = ["OP6", "OP5", "OP4", "OP3", "OP2", "OP1"]
    parameters = {}
    
    for idx, op in enumerate(operators):
        base_offset = idx * 17
        parameters.update({
            f"{op} EG rate 1": read_byte(base_offset),
            f"{op} EG rate 2": read_byte(base_offset + 1),
            f"{op} EG rate 3": read_byte(base_offset + 2),
            f"{op} EG rate 4": read_byte(base_offset + 3),
            f"{op} EG level 1": read_byte(base_offset + 4),
            f"{op} EG level 2": read_byte(base_offset + 5),
            f"{op} EG level 3": read_byte(base_offset + 6),
            f"{op} EG level 4": read_byte(base_offset + 7),
            f"{op} KBD LEV SCL BRK PT": read_byte(base_offset + 8),
            f"{op} KBD LEV SCL LFT DEPTH": read_byte(base_offset + 9),
            f"{op} KBD LEV SCL RHT DEPTH": read_byte(base_offset + 10),
            f"{op} KBD LEV SCL LFT CURVE": read_byte(base_offset + 11, 0b00000011),
            f"{op} KBD LEV SCL RHT CURVE": read_byte(base_offset + 11, 0b00001100, 2),
            f"{op} OSC DETUNE": read_byte(base_offset + 12, 0b00000111),
            f"{op} KBD RATE SCALING": read_byte(base_offset + 12, 0b00111000, 3),
            f"{op} KEY VEL SENSITIVITY": read_byte(base_offset + 13, 0b00111000, 3),
            f"{op} AMP MOD SENSITIVITY": read_byte(base_offset + 13, 0b00000011),
            f"{op} OPERATOR OUTPUT LEVEL": read_byte(base_offset + 14),
            f"{op} OSC MODE": read_byte(base_offset + 15, 0b00000001),
            f"{op} OSC FREQ COARSE": read_byte(base_offset + 15, 0b01111110, 1),
            f"{op} OSC FREQ FINE": read_byte(base_offset + 16)
        })
    
    parameters.update({
        "PITCH EG rate 1": read_byte(102),
        "PITCH EG rate 2": read_byte(103),
        "PITCH EG rate 3": read_byte(104),
        "PITCH EG rate 4": read_byte(105),
        "PITCH EG level 1": read_byte(106),
        "PITCH EG level 2": read_byte(107),
        "PITCH EG level 3": read_byte(108),
        "PITCH EG level 4": read_byte(109),
        "ALGORITHM #": read_byte(110, 0b00111111),
        "OSCILLATOR SYNC": read_byte(111, 0b00000100, 2),
        "FEEDBACK": read_byte(111, 0b00000011),
        "LFO SPEED": read_byte(112),
        "LFO DELAY": read_byte(113),
        "LFO PITCH MOD DEPTH": read_byte(114),
        "LFO AMP MOD DEPTH": read_byte(115),
        "LFO SYNC": read_byte(116, 0b00000001),
        "LFO WAVEFORM": read_byte(116, 0b00111000, 3),
        "PITCH MOD SENSITIVITY": read_byte(116, 0b01110000, 4),
        "TRANSPOSE": read_byte(117),
        "VOICE NAME": ''.join([read_char(i) for i in range(118, 128)])
    })
    return parameters


def generate_dataset(fpath):
    # Reading the binary data
    with open(fpath, 'rb') as f:
        data = f.read()

    # Generating the dataset
    patches = [data[i:i+128] for i in range(0, len(data), 128)]
    dataset = [unpack_patch(patch) for patch in patches]

    return dataset


def pack_patch(parameters):
    packed_data = bytearray(128)  # Initialize with 128 zeros

    def write_byte(offset, value):
        packed_data[offset] = value & 0x7F

    def pack_byte(high_nibble, low_nibble):
        return ((high_nibble << 4) & 0xF0) | (low_nibble & 0x0F)

    # Pack the operators
    operators = ["OP6", "OP5", "OP4", "OP3", "OP2", "OP1"]
    for idx, op in enumerate(operators):
        base_offset = idx * 17
        write_byte(base_offset, parameters[f"{op} EG rate 1"])
        write_byte(base_offset + 1, parameters[f"{op} EG rate 2"])
        write_byte(base_offset + 2, parameters[f"{op} EG rate 3"])
        write_byte(base_offset + 3, parameters[f"{op} EG rate 4"])
        write_byte(base_offset + 4, parameters[f"{op} EG level 1"])
        write_byte(base_offset + 5, parameters[f"{op} EG level 2"])
        write_byte(base_offset + 6, parameters[f"{op} EG level 3"])
        write_byte(base_offset + 7, parameters[f"{op} EG level 4"])
        write_byte(base_offset + 8, parameters[f"{op} KBD LEV SCL BRK PT"])
        write_byte(base_offset + 9, parameters[f"{op} KBD LEV SCL LFT DEPTH"])
        write_byte(base_offset + 10, parameters[f"{op} KBD LEV SCL RHT DEPTH"])
        packed_data[base_offset + 11] = pack_byte(parameters[f"{op} KBD LEV SCL RHT CURVE"], parameters[f"{op} KBD LEV SCL LFT CURVE"])
        packed_data[base_offset + 12] = pack_byte(parameters[f"{op} KBD RATE SCALING"], parameters[f"{op} OSC DETUNE"])
        packed_data[base_offset + 13] = pack_byte(parameters[f"{op} KEY VEL SENSITIVITY"], parameters[f"{op} AMP MOD SENSITIVITY"])
        write_byte(base_offset + 14, parameters[f"{op} OPERATOR OUTPUT LEVEL"])
        packed_data[base_offset + 15] = ((parameters[f"{op} OSC MODE"] << 6) & 0x40) | (parameters[f"{op} OSC FREQ COARSE"] & 0x3F)
        write_byte(base_offset + 16, parameters[f"{op} OSC FREQ FINE"])

    # After the operators, pack the rest of the parameters
    write_byte(102, parameters["PITCH EG rate 1"])
    write_byte(103, parameters["PITCH EG rate 2"])
    write_byte(104, parameters["PITCH EG rate 3"])
    write_byte(105, parameters["PITCH EG rate 4"])
    write_byte(106, parameters["PITCH EG level 1"])
    write_byte(107, parameters["PITCH EG level 2"])
    write_byte(108, parameters["PITCH EG level 3"])
    write_byte(109, parameters["PITCH EG level 4"])
    packed_data[110] = parameters["ALGORITHM #"] & 0x3F
    packed_data[111] = pack_byte(parameters["FEEDBACK"], parameters["OSCILLATOR SYNC"])
    write_byte(112, parameters["LFO SPEED"])
    write_byte(113, parameters["LFO DELAY"])
    write_byte(114, parameters["LFO PITCH MOD DEPTH"])
    write_byte(115, parameters["LFO AMP MOD DEPTH"])
    packed_data[116] = ((parameters["PITCH MOD SENSITIVITY"] << 4) & 0x70) | (parameters["LFO WAVEFORM"] << 1) | parameters["LFO SYNC"]
    write_byte(117, parameters["TRANSPOSE"])

    # Handle the voice name
    voice_name = parameters.get("VOICE NAME", "")  # Fetch the name or use an empty string if it's missing
    voice_name_padded = voice_name.ljust(10)  # Pad with spaces to make it 10 characters
    for i in range(10):
        write_byte(118 + i, ord(voice_name_padded[i]))

    return packed_data

def generate_sysex_with_corrected_checksum(data):
    # Create a SysEx message for 32 patches
    sysex_start = bytes([0xF0, 0x43, 0x00, 0x09, 0x20, 0x00])
    checksum = (128 - sum(data) % 128) % 128
    sysex_end = bytes([checksum, 0xF7])

    return sysex_start + data + sysex_end

def write_sysex_to_file(sysex_data, filename):
    with open(filename, 'wb') as sysex_file:
        sysex_file.write(sysex_data)

def example():
    # Load the dataset from compact.bin
    dataset = generate_dataset("data/compact.bin")  # Replace with the actual path to compact.bin

    # Select 32 random patches
    random_patches = random.sample(dataset, 32)
    
    # Remove voice names
    for patch in random_patches:
        if "VOICE NAME" in patch:
            del patch["VOICE NAME"]

    # Pack the selected patches
    packed_patches = bytearray()
    for patch in random_patches:
        packed_patches += pack_patch(patch)

    # Generate the SysEx message for the selected patches
    sysex_data_32_patches = generate_sysex_with_corrected_checksum(packed_patches)

    # Write the SysEx message to a file
    write_sysex_to_file(sysex_data_32_patches, "random_patches_output.syx")


if __name__ == '__main__':
    example()