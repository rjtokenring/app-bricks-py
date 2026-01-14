package llamacppwrapper

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/cavaliergopher/grab/v3"
)

func DownloadMode(mode, model string) error {
	switch mode {
	case "ollama":
		return downloadOllamaHostedModel(model)
	default:
		return fmt.Errorf("unsupported download mode: %s", mode)
	}
}

type layer struct {
	MediaType string `json:"mediaType"`
	Size      int64  `json:"size"`
	Digest    string `json:"digest"`
}

type ollamaLayer struct {
	Layers []layer `json:"layers"`
}

func downloadOllamaHostedModel(model string) error {
	if !strings.Contains(model, ":") {
		return fmt.Errorf("invalid model format for ollama, expected 'model:version'")
	}

	parts := strings.SplitN(model, ":", 2)
	modelName := strings.TrimSpace(parts[0])
	version := strings.TrimSpace(parts[1])
	slog.Info("Downloading Ollama model", "model", modelName, "version", version)
	// Get layers for the model
	if resp, err := http.Get(fmt.Sprintf("https://registry.ollama.ai/v2/library/%s/manifests/%s", modelName, version)); err != nil {
		return err
	} else {
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("failed to get model manifest, status code: %d", resp.StatusCode)
		}
		// Here you would parse the manifest and download the layers accordingly.
		// For brevity, we'll skip the actual download implementation.
		slog.Info("Successfully fetched model manifest", "model", modelName, "version", version)

		bytesBuffer := new(bytes.Buffer)
		bytesBuffer.ReadFrom(resp.Body)
		slog.Debug("Manifest content", "content", bytesBuffer.String())

		layers := ollamaLayer{}
		if err = json.Unmarshal(bytesBuffer.Bytes(), &layers); err != nil {
			return err
		}

		// Now you would iterate over layers and download the actual model files.
		for _, layer := range layers.Layers {
			if layer.MediaType == "application/vnd.ollama.image.model" {
				slog.Info("Found model layer", "digest", layer.Digest, "size", layer.Size)
				// Download the layer here
				downloadUrl := fmt.Sprintf("https://registry.ollama.ai/v2/library/%s/blobs/%s", modelName, layer.Digest)
				if downRequest, err := grab.NewRequest(fmt.Sprintf("%s-%s.gguf", modelName, version), downloadUrl); err != nil {
					return err
				} else {
					slog.Info("Starting download", "url", downloadUrl)
					resp := grab.DefaultClient.Do(downRequest)
					if resp.DidResume {
						slog.Info("Resumed previous download", "filename", resp.Filename)
					}

					// User feedback loop
					t := time.NewTicker(2000 * time.Millisecond)
					defer t.Stop()

					for {
						select {
						case <-t.C:
							fmt.Printf("  transferred %v / %v bytes (%.2f%%)\n",
								resp.BytesComplete(),
								resp.Size(),
								100*resp.Progress())

						case <-resp.Done:
							// Download is finished
							fmt.Printf("Download saved to %v\n", resp.Filename)
							return nil
						}
					}
				}
			}
		}

		return nil
	}
}
