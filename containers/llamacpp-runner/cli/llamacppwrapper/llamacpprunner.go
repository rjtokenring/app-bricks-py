package llamacppwrapper

import (
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"syscall"
)

func DownloadMode(model string) (int, error) {
	if inf, err := os.Stat(fmt.Sprintf("%s.partial", model)); err == nil && inf.Size() > 0 {
		fmt.Println("Resuming partial download...")
	}

	cmd := exec.Command("llama-run", model, "-ngl", "16", "\"1+1=?\"")

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		panic(err)
	}
	cmd.Stderr = cmd.Stdout

	if err := cmd.Start(); err != nil {
		panic(err)
	}

	go func() {
		whitespaceRe := regexp.MustCompile(`\s+`)
		fullCharRe := regexp.MustCompile(`[^\x00-\x7F]+`)
		buf := make([]byte, 1024)
		for {
			n, err := stdoutPipe.Read(buf)
			if n > 0 {
				out := whitespaceRe.ReplaceAll(buf[:n], []byte(""))
				out = fullCharRe.ReplaceAll(out, []byte(""))
				line := string(out)
				line = strings.Trim(line, " ")
				if line == "" {
					continue
				}
				fmt.Println("-----------")
				fmt.Println(line)
			}
			if err != nil {
				break
			}
		}
	}()

	// Wait for command to finish
	err = cmd.Wait()

	// Get exit code
	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			if status, ok := exitErr.Sys().(syscall.WaitStatus); ok {
				exitCode = status.ExitStatus()
			}
		} else {
			fmt.Println("Error running command:", err)
		}
	} else {
		if status, ok := cmd.ProcessState.Sys().(syscall.WaitStatus); ok {
			exitCode = status.ExitStatus()
		}
	}

	fmt.Println("Process finished with exit code:", exitCode)

	return exitCode, nil
}
