// Pure-nodes ComfyUI workflow builder (the Revise / 'copy as nodes' graph).
// Kept in its own file so tests/js/comfynodes_parity.test.js can pin comfy_nodes_workflow.py to it.

// Build a pure ComfyUI nodes workflow JSON from image config data
// Reusable version of the clipboard copy logic — returns the workflow object
function buildComfyNodesWorkflow(d) {
    // Parse LoRA string into array
    const loras = [];
    if (d.lora && d.lora !== "None") {
        // Fix: Filter empty entries to prevent 'ghost' nodes from trailing " + "
        const loraEntries = d.lora.split(' + ').filter(e => e.trim().length > 0);
        loraEntries.forEach(entry => {
            const parts = entry.split(':');
            const name = parts[0];
            const strength_model = parseFloat(parts[1] || 1.0);
            const strength_clip = parseFloat(parts[2] || strength_model);
            loras.push({ name, strength_model, strength_clip });
        });
    }

    // Generate node IDs
    let nodeId = 1;
    const checkpointNode = nodeId++;
    const loraNodes = loras.map(() => nodeId++);
    const positiveClipNode = nodeId++;
    const negativeClipNode = nodeId++;
    const emptyLatentNode = nodeId++;
    const ksamplerNode = nodeId++;
    const vaeDecodeNode = nodeId++;
    const previewNode = nodeId++;

    if (!crypto.randomUUID) {
        crypto.randomUUID = function () {
            return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, c =>
                (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
            );
        };
    }

    // Build the workflow JSON
    const workflow = {
        id: crypto.randomUUID(),
        revision: 0,
        last_node_id: nodeId - 1,
        last_link_id: 100, // We will update this at the end
        nodes: [],
        links: [],
        groups: [],
        config: {},
        extra: {
            workflowRendererVersion: "LG",
            ds: { scale: 0.573, offset: [488, 377] }
        },
        version: 0.4
    };

    // --- 1. CREATE NODES ---
    // (This section remains mostly the same, just ensured clean initialization)

    // Checkpoint
    workflow.nodes.push({
        id: checkpointNode,
        type: "CheckpointLoaderSimple",
        pos: [-200, 60],
        size: [315, 98],
        flags: {}, order: 0, mode: 0,
        inputs: [],
        outputs: [
            { name: "MODEL", type: "MODEL", links: [] },
            { name: "CLIP", type: "CLIP", links: [] },
            { name: "VAE", type: "VAE", links: [] }
        ],
        properties: { "Node name for S&R": "CheckpointLoaderSimple" },
        widgets_values: [d.model || "XL\\waiANIPONYXL_v140_fp8_e4m3fn_full.safetensors"]
    });

    // LoRAs
    loras.forEach((lora, index) => {
        workflow.nodes.push({
            id: loraNodes[index],
            type: "LoraLoader",
            pos: [170 + (index * 312), 60],
            size: [270, 126],
            flags: {}, order: index + 1, mode: 0,
            inputs: [
                { name: "model", type: "MODEL", link: null },
                { name: "clip", type: "CLIP", link: null }
            ],
            outputs: [
                { name: "MODEL", type: "MODEL", links: [] },
                { name: "CLIP", type: "CLIP", links: [] }
            ],
            properties: { "Node name for S&R": "LoraLoader" },
            widgets_values: [String(lora.name).replace(/\//g, "\\"), lora.strength_model, lora.strength_clip]
        });
    });

    const posX = 910 + (loras.length * 312);

    // Positive Clip
    workflow.nodes.push({
        id: positiveClipNode,
        type: "CLIPTextEncode",
        pos: [posX, 3.52],
        size: [460, 190],
        flags: {}, order: loras.length + 1, mode: 0,
        inputs: [{ name: "clip", type: "CLIP", link: null }],
        outputs: [{ name: "CONDITIONING", type: "CONDITIONING", links: [] }],
        properties: { "Node name for S&R": "CLIPTextEncode" },
        widgets_values: [d.positive || ""]
    });

    // Negative Clip
    workflow.nodes.push({
        id: negativeClipNode,
        type: "CLIPTextEncode",
        pos: [posX, 240],
        size: [470, 200],
        flags: {}, order: loras.length + 2, mode: 0,
        inputs: [{ name: "clip", type: "CLIP", link: null }],
        outputs: [{ name: "CONDITIONING", type: "CONDITIONING", links: [] }],
        properties: { "Node name for S&R": "CLIPTextEncode" },
        widgets_values: [d.negative || ""]
    });

    // Empty Latent
    workflow.nodes.push({
        id: emptyLatentNode,
        type: "EmptyLatentImage",
        pos: [posX + 130, 510],
        size: [270, 106],
        flags: {}, order: 1, mode: 0,
        inputs: [],
        outputs: [{ name: "LATENT", type: "LATENT", links: [] }],
        properties: { "Node name for S&R": "EmptyLatentImage" },
        widgets_values: [d.width || 1080, d.height || 1584, 1]
    });

    // KSampler
    const sampX = posX + 510;
    workflow.nodes.push({
        id: ksamplerNode,
        type: "KSampler",
        pos: [sampX, 23.52],
        size: [315, 708],
        flags: {}, order: loras.length + 3, mode: 0,
        inputs: [
            { name: "model", type: "MODEL", link: null },
            { name: "positive", type: "CONDITIONING", link: null },
            { name: "negative", type: "CONDITIONING", link: null },
            { name: "latent_image", type: "LATENT", link: null }
        ],
        outputs: [{ name: "LATENT", type: "LATENT", links: [] }],
        properties: { "Node name for S&R": "KSampler" },
        widgets_values: [
            d.seed || 0, "fixed", d.steps || 25, d.cfg || 7,
            d.sampler || "dpmpp_2m", d.scheduler || "karras", d.denoise || 1
        ]
    });

    // VAE Decode
    workflow.nodes.push({
        id: vaeDecodeNode,
        type: "VAEDecode",
        pos: [sampX + 390, 33],
        size: [210, 46],
        flags: {}, order: loras.length + 4, mode: 0,
        inputs: [
            { name: "samples", type: "LATENT", link: null },
            { name: "vae", type: "VAE", link: null }
        ],
        outputs: [{ name: "IMAGE", type: "IMAGE", links: [] }],
        properties: { "Node name for S&R": "VAEDecode" },
        widgets_values: []
    });

    // Preview
    workflow.nodes.push({
        id: previewNode,
        type: "PreviewImage",
        pos: [sampX + 370, 173],
        size: [418, 556],
        flags: {}, order: loras.length + 5, mode: 0,
        inputs: [{ name: "images", type: "IMAGE", link: null }],
        outputs: [],
        properties: { "Node name for S&R": "PreviewImage" },
        widgets_values: []
    });


    // --- 2. WIRE NODES (THE FIX) ---

    const getNode = (id) => workflow.nodes.find(n => n.id === id);
    let linkId = 1;

    // TRACKERS: These hold the ID of the node currently supplying the signal
    // We start with the Checkpoint
    let currentModelSource = { id: checkpointNode, slot: 0 };
    let currentClipSource = { id: checkpointNode, slot: 1 };

    // 1. Loop through LoRAs and Daisy Chain them
    // Checkpoint -> Lora1 -> Lora2 -> ...
    for (let i = 0; i < loras.length; i++) {
        const thisLoraId = loraNodes[i];

        // Wire MODEL: Previous Output -> This Lora Input
        workflow.links.push([linkId, currentModelSource.id, currentModelSource.slot, thisLoraId, 0, "MODEL"]);
        getNode(currentModelSource.id).outputs[currentModelSource.slot].links.push(linkId);
        getNode(thisLoraId).inputs[0].link = linkId;
        linkId++;

        // Wire CLIP: Previous Output -> This Lora Input
        workflow.links.push([linkId, currentClipSource.id, currentClipSource.slot, thisLoraId, 1, "CLIP"]);
        getNode(currentClipSource.id).outputs[currentClipSource.slot].links.push(linkId);
        getNode(thisLoraId).inputs[1].link = linkId;
        linkId++;

        // Update Trackers: The output of THIS Lora is now the source for the next step
        currentModelSource = { id: thisLoraId, slot: 0 };
        currentClipSource = { id: thisLoraId, slot: 1 };
    }

    // 2. Connect Final Signal (from last LoRA or Checkpoint) to Engines

    // Final Model -> KSampler
    workflow.links.push([linkId, currentModelSource.id, currentModelSource.slot, ksamplerNode, 0, "MODEL"]);
    getNode(currentModelSource.id).outputs[currentModelSource.slot].links.push(linkId);
    getNode(ksamplerNode).inputs[0].link = linkId;
    linkId++;

    // Final CLIP -> Positive Prompt
    workflow.links.push([linkId, currentClipSource.id, currentClipSource.slot, positiveClipNode, 0, "CLIP"]);
    getNode(currentClipSource.id).outputs[currentClipSource.slot].links.push(linkId);
    getNode(positiveClipNode).inputs[0].link = linkId;
    linkId++;

    // Final CLIP -> Negative Prompt (Shared link logic, but new link ID for Comfy)
    workflow.links.push([linkId, currentClipSource.id, currentClipSource.slot, negativeClipNode, 0, "CLIP"]);
    getNode(currentClipSource.id).outputs[currentClipSource.slot].links.push(linkId);
    getNode(negativeClipNode).inputs[0].link = linkId;
    linkId++;

    // 3. Connect the rest of the standard components

    // VAE: Checkpoint -> VAE Decode
    workflow.links.push([linkId, checkpointNode, 2, vaeDecodeNode, 1, "VAE"]);
    getNode(checkpointNode).outputs[2].links.push(linkId);
    getNode(vaeDecodeNode).inputs[1].link = linkId;
    linkId++;

    // Conditioning: Positive -> KSampler
    workflow.links.push([linkId, positiveClipNode, 0, ksamplerNode, 1, "CONDITIONING"]);
    getNode(positiveClipNode).outputs[0].links.push(linkId);
    getNode(ksamplerNode).inputs[1].link = linkId;
    linkId++;

    // Conditioning: Negative -> KSampler
    workflow.links.push([linkId, negativeClipNode, 0, ksamplerNode, 2, "CONDITIONING"]);
    getNode(negativeClipNode).outputs[0].links.push(linkId);
    getNode(ksamplerNode).inputs[2].link = linkId;
    linkId++;

    // Latent: Empty Latent -> KSampler
    workflow.links.push([linkId, emptyLatentNode, 0, ksamplerNode, 3, "LATENT"]);
    getNode(emptyLatentNode).outputs[0].links.push(linkId);
    getNode(ksamplerNode).inputs[3].link = linkId;
    linkId++;

    // Latent: KSampler -> VAE Decode
    workflow.links.push([linkId, ksamplerNode, 0, vaeDecodeNode, 0, "LATENT"]);
    getNode(ksamplerNode).outputs[0].links.push(linkId);
    getNode(vaeDecodeNode).inputs[0].link = linkId;
    linkId++;

    // Image: VAE Decode -> Preview
    workflow.links.push([linkId, vaeDecodeNode, 0, previewNode, 0, "IMAGE"]);
    getNode(vaeDecodeNode).outputs[0].links.push(linkId);
    getNode(previewNode).inputs[0].link = linkId;
    linkId++;

    // Update workflow config
    workflow.last_link_id = linkId;
    return workflow;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { buildComfyNodesWorkflow: buildComfyNodesWorkflow };
}
