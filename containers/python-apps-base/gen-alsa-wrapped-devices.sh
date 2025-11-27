#!/bin/bash

# Generate ALSA config for multiple cards upfront

ASOUND_CONF="$HOME/.asoundrc"
MAX_CARDS=10  # Adjust as needed

cat > "$ASOUND_CONF" << 'EOF'
# Auto-generated ALSA configurations for sound cards 0-9
EOF

for CARD_NUM in $(seq 0 $((MAX_CARDS - 1))); do
cat >> "$ASOUND_CONF" << EOF

# Configuration for card $CARD_NUM
pcm.card_${CARD_NUM}_dmix {
    type dmix
    ipc_key $((1024 + CARD_NUM))
    ipc_key_add_uid true
    slave {
        pcm "hw:${CARD_NUM},0"
    }
}

pcm.card_${CARD_NUM}_plug_dmix {
    type plug
    slave.pcm "card_${CARD_NUM}_dmix"
}

pcm.card_${CARD_NUM}_spk_wr {
    type softvol
    slave {
        pcm "card_${CARD_NUM}_plug_dmix"
    }
    control {
        name "card_${CARD_NUM}_spk_wr"
        card ${CARD_NUM}
    }
}
EOF
done

for CARD_NUM in $(seq 0 $((MAX_CARDS - 1))); do
cat >> "$ASOUND_CONF" << EOF

# Configuration for card $CARD_NUM
pcm.card_${CARD_NUM}_dsnoop {
    type dsnoop
    ipc_key $((1124 + CARD_NUM))
    ipc_key_add_uid true
    slave {
        pcm "hw:${CARD_NUM},0"
    }
}

pcm.card_${CARD_NUM}_softvol_dsnoop {
    type softvol
    slave.pcm "card_${CARD_NUM}_dsnoop"
    control {
        name "card_${CARD_NUM}_mic_wr"
        card ${CARD_NUM}
    }    
}

pcm.card_${CARD_NUM}_mic_wr {
    type plug
    slave {
        pcm "card_${CARD_NUM}_softvol_dsnoop"
    }
}
EOF
done