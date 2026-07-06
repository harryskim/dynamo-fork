/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package setup

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
)

func TestSetupAllRequiresCompleteOptions(t *testing.T) {
	tests := []struct {
		name    string
		opts    Options
		wantErr string
	}{
		{
			name:    "operator configuration",
			wantErr: "operator configuration is required",
		},
		{
			name:    "runtime configuration",
			opts:    Options{Config: &configv1alpha1.OperatorConfiguration{}},
			wantErr: "runtime configuration is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := SetupAll(nil, tt.opts)
			if err == nil || err.Error() != tt.wantErr {
				t.Fatalf("SetupAll() error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}
