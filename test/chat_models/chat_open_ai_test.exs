defmodule LangChain.ChatModels.ChatOpenAITest do
  use LangChain.BaseCase
  import LangChain.Fixtures

  doctest LangChain.ChatModels.ChatOpenAI
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Function
  alias LangChain.FunctionParam

  @test_model "gpt-3.5-turbo"

  defp hello_world(_args, _context) do
    "Hello world!"
  end

  setup do
    {:ok, hello_world} =
      Function.new(%{
        name: "hello_world",
        description: "Give a hello world greeting.",
        function: fn -> IO.puts("Hello world!") end
      })

    %{hello_world: hello_world}
  end

  describe "new/1" do
    test "works with minimal attr" do
      assert {:ok, %ChatOpenAI{} = openai} = ChatOpenAI.new(%{"model" => @test_model})
      assert openai.model == @test_model
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatOpenAI.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:1234/v1/chat/completions"

      model =
        ChatOpenAI.new!(%{
          endpoint: override_url
        })

      assert model.endpoint == override_url
    end
  end

  describe "for_api/3" do
    test "generates a map for an API call" do
      {:ok, openai} =
        ChatOpenAI.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "frequency_penalty" => 0.5,
          "api_key" => "api_key"
        })

      data = ChatOpenAI.for_api(openai, [], [])
      assert data.model == @test_model
      assert data.temperature == 1
      assert data.frequency_penalty == 0.5
      assert data.response_format == %{"type" => "text"}
    end

    test "generates a map for an API call with JSON response set to true" do
      {:ok, openai} =
        ChatOpenAI.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "frequency_penalty" => 0.5,
          "json_response" => true
        })

      data = ChatOpenAI.for_api(openai, [], [])
      assert data.model == @test_model
      assert data.temperature == 1
      assert data.frequency_penalty == 0.5
      assert data.response_format == %{"type" => "json_object"}
    end

    test "generates a map for an API call with max_tokens set" do
      {:ok, openai} =
        ChatOpenAI.new(%{
          "model" => @test_model,
          "temperature" => 1,
          "frequency_penalty" => 0.5,
          "max_tokens" => 1234
        })

      data = ChatOpenAI.for_api(openai, [], [])
      assert data.model == @test_model
      assert data.temperature == 1
      assert data.frequency_penalty == 0.5
      assert data.max_tokens == 1234
    end
  end

  describe "for_api/1" do
    test "turns a function_call into expected JSON format" do
      msg = Message.new_function_call!("hello_world", "{}")

      json = ChatOpenAI.for_api(msg)

      assert json == %{
               "content" => nil,
               "function_call" => %{"arguments" => "{}", "name" => "hello_world"},
               "role" => :assistant
             }
    end

    test "turns a function_call into expected JSON format with arguments" do
      args = %{"expression" => "11 + 10"}
      msg = Message.new_function_call!("hello_world", Jason.encode!(args))

      json = ChatOpenAI.for_api(msg)

      assert json == %{
               "content" => nil,
               "function_call" => %{
                 "arguments" => "{\"expression\":\"11 + 10\"}",
                 "name" => "hello_world"
               },
               "role" => :assistant
             }
    end

    test "turns a function response into expected JSON format" do
      msg = Message.new_function!("hello_world", "Hello World!")

      json = ChatOpenAI.for_api(msg)

      assert json == %{"content" => "Hello World!", "name" => "hello_world", "role" => :function}
    end

    test "works with minimal definition and no parameters" do
      {:ok, fun} = Function.new(%{"name" => "hello_world"})

      result = ChatOpenAI.for_api(fun)
      # result = Function.for_api(fun)

      assert result == %{
               "name" => "hello_world",
               #  NOTE: Sends the required empty parameter definition when none set
               "parameters" => %{"properties" => %{}, "type" => "object"}
             }
    end

    test "supports parameters" do
      params_def = %{
        "type" => "object",
        "properties" => %{
          "p1" => %{"type" => "string"},
          "p2" => %{"description" => "Param 2", "type" => "number"},
          "p3" => %{
            "enum" => ["yellow", "red", "green"],
            "type" => "string"
          }
        },
        "required" => ["p1"]
      }

      {:ok, fun} =
        Function.new(%{
          name: "say_hi",
          description: "Provide a friendly greeting.",
          parameters: [
            FunctionParam.new!(%{name: "p1", type: :string, required: true}),
            FunctionParam.new!(%{name: "p2", type: :number, description: "Param 2"}),
            FunctionParam.new!(%{name: "p3", type: :string, enum: ["yellow", "red", "green"]})
          ]
        })

      # result = Function.for_api(fun)
      result = ChatOpenAI.for_api(fun)

      assert result == %{
               "name" => "say_hi",
               "description" => "Provide a friendly greeting.",
               "parameters" => params_def
             }
    end

    test "supports parameters_schema" do
      params_def = %{
        "type" => "object",
        "properties" => %{
          "p1" => %{"description" => nil, "type" => "string"},
          "p2" => %{"description" => "Param 2", "type" => "number"},
          "p3" => %{
            "description" => nil,
            "enum" => ["yellow", "red", "green"],
            "type" => "string"
          }
        },
        "required" => ["p1"]
      }

      {:ok, fun} =
        Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters_schema" => params_def
        })

      # result = Function.for_api(fun)
      result = ChatOpenAI.for_api(fun)

      assert result == %{
               "name" => "say_hi",
               "description" => "Provide a friendly greeting.",
               "parameters" => params_def
             }
    end

    test "does not allow both parameters and parameters_schema" do
      {:error, changeset} =
        Function.new(%{
          name: "problem",
          parameters: [
            FunctionParam.new!(%{name: "p1", type: :string, required: true})
          ],
          parameters_schema: %{stuff: true}
        })

      assert {"Cannot use both parameters and parameters_schema", _} =
               changeset.errors[:parameters]
    end

    test "does not include the function to execute" do
      # don't try and send an Elixir function ref through to the API
      {:ok, fun} = Function.new(%{"name" => "hello_world", "function" => &hello_world/2})
      # result = Function.for_api(fun)
      result = ChatOpenAI.for_api(fun)
      refute Map.has_key?(result, "function")
    end
  end

  describe "call/2" do
    @tag live_call: true, live_open_ai: true
    test "basic content example" do
      # set_fake_llm_response({:ok, Message.new_assistant("\n\nRainbow Sox Co.")})

      # https://js.langchain.com/docs/modules/models/chat/
      {:ok, chat} = ChatOpenAI.new(%{temperature: 1, seed: 0})

      {:ok, [%Message{role: :assistant, content: response}]} =
        ChatOpenAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert response =~ "Colorful Threads"
    end

    @tag live_call: true, live_open_ai: true
    test "basic streamed content example's final result" do
      # set_fake_llm_response({:ok, Message.new_assistant("\n\nRainbow Sox Co.")})

      # https://js.langchain.com/docs/modules/models/chat/
      {:ok, chat} = ChatOpenAI.new(%{temperature: 1, seed: 0, stream: true})

      {:ok, result} =
        ChatOpenAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      # returns a list of MessageDeltas. A list of a list because it's "n" choices.
      assert result == [
               [
                 %LangChain.MessageDelta{
                   content: "",
                   status: :incomplete,
                   index: 0,
                   function_name: nil,
                   role: :assistant,
                   arguments: nil
                 }
               ],
               [
                 %LangChain.MessageDelta{
                   content: "Color",
                   status: :incomplete,
                   index: 0,
                   function_name: nil,
                   role: :unknown,
                   arguments: nil
                 }
               ],
               [
                 %LangChain.MessageDelta{
                   content: "ful",
                   status: :incomplete,
                   index: 0,
                   function_name: nil,
                   role: :unknown,
                   arguments: nil
                 }
               ],
               [
                 %LangChain.MessageDelta{
                   content: " Threads",
                   status: :incomplete,
                   index: 0,
                   function_name: nil,
                   role: :unknown,
                   arguments: nil
                 }
               ],
               [
                 %LangChain.MessageDelta{
                   content: nil,
                   status: :complete,
                   index: 0,
                   function_name: nil,
                   role: :unknown,
                   arguments: nil
                 }
               ]
             ]
    end

    @tag live_call: true, live_open_ai: true
    test "executing a function", %{hello_world: hello_world} do
      {:ok, chat} = ChatOpenAI.new(%{seed: 0})

      {:ok, message} =
        Message.new_user("Only using the functions you have been provided with, give a greeting.")

      {:ok, [message]} = ChatOpenAI.call(chat, [message], [hello_world])

      assert %Message{role: :assistant} = message
      assert message.arguments == %{}
      assert message.content == nil
    end

    @tag live_call: true, live_open_ai: true
    test "executes callback function when data is streamed" do
      callback = fn %MessageDelta{} = delta ->
        send(self(), {:message_delta, delta})
      end

      # https://js.langchain.com/docs/modules/models/chat/
      {:ok, chat} = ChatOpenAI.new(%{seed: 0, temperature: 1, stream: true})

      {:ok, _post_results} =
        ChatOpenAI.call(
          chat,
          [
            Message.new_user!("Return the exact response 'Hi'.")
          ],
          [],
          callback
        )

      # we expect to receive the response over 3 delta messages
      assert_receive {:message_delta, delta_1}, 500
      assert_receive {:message_delta, delta_2}, 500
      assert_receive {:message_delta, delta_3}, 500

      # IO.inspect(delta_1)
      # IO.inspect(delta_2)
      # IO.inspect(delta_3)

      merged =
        delta_1
        |> MessageDelta.merge_delta(delta_2)
        |> MessageDelta.merge_delta(delta_3)

      assert merged.role == :assistant
      assert merged.content =~ "Hi"
      assert merged.status == :complete
    end

    @tag live_call: true, live_open_ai: true
    test "executes callback function when data is NOT streamed" do
      callback = fn %Message{} = new_message ->
        send(self(), {:message_received, new_message})
      end

      # https://js.langchain.com/docs/modules/models/chat/
      # NOTE streamed. Should receive complete message.
      {:ok, chat} = ChatOpenAI.new(%{seed: 0, temperature: 1, stream: false})

      {:ok, [message]} =
        ChatOpenAI.call(
          chat,
          [
            Message.new_user!("Return the response 'Hi'.")
          ],
          [],
          callback
        )

      assert message.content =~ "Hi"
      assert message.index == 0
      assert_receive {:message_received, received_item}, 500
      assert %Message{} = received_item
      assert received_item.role == :assistant
      assert received_item.content =~ "Hi"
      assert received_item.index == 0
    end

    @tag live_call: true, live_open_ai: true
    test "handles when request is too large" do
      {:ok, chat} =
        ChatOpenAI.new(%{model: "gpt-3.5-turbo-0301", seed: 0, stream: false, temperature: 1})

      {:error, reason} = ChatOpenAI.call(chat, [too_large_user_request()])
      assert reason =~ "maximum context length"
    end
  end

  describe "do_process_response/1" do
    test "handles receiving a message" do
      response = %{
        "message" => %{"role" => "assistant", "content" => "Greetings!"},
        "finish_reason" => "stop",
        "index" => 1
      }

      assert %Message{} = struct = ChatOpenAI.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == "Greetings!"
      assert struct.index == 1
    end

    test "handles receiving a function_call message" do
      response = %{
        "finish_reason" => "function_call",
        "index" => 0,
        "message" => %{
          "content" => nil,
          "function_call" => %{"arguments" => "{}", "name" => "hello_world"},
          "role" => "assistant"
        }
      }

      assert %Message{} = struct = ChatOpenAI.do_process_response(response)

      assert struct.role == :assistant
      assert struct.content == nil
      assert struct.function_name == "hello_world"
      assert struct.arguments == %{}
      assert struct.index == 0
    end

    test "handles error from server that the max length has been reached" do
      response = %{
        "finish_reason" => "length",
        "index" => 0,
        "message" => %{
          "content" => "Some of the response that was abruptly",
          "role" => "assistant"
        }
      }

      assert %Message{} = struct = ChatOpenAI.do_process_response(response)

      assert struct.role == :assistant
      assert struct.content == "Some of the response that was abruptly"
      assert struct.index == 0
      assert struct.status == :length
    end

    test "handles receiving a delta message for a content message at different parts" do
      delta_content = LangChain.Fixtures.raw_deltas_for_content()

      msg_1 = Enum.at(delta_content, 0)
      msg_2 = Enum.at(delta_content, 1)
      msg_10 = Enum.at(delta_content, 10)

      expected_1 = %MessageDelta{
        content: "",
        index: 0,
        function_name: nil,
        role: :assistant,
        arguments: nil,
        status: :incomplete
      }

      [%MessageDelta{} = delta_1] = ChatOpenAI.do_process_response(msg_1)
      assert delta_1 == expected_1

      expected_2 = %MessageDelta{
        content: "Hello",
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        status: :incomplete
      }

      [%MessageDelta{} = delta_2] = ChatOpenAI.do_process_response(msg_2)
      assert delta_2 == expected_2

      expected_10 = %MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        status: :complete
      }

      [%MessageDelta{} = delta_10] = ChatOpenAI.do_process_response(msg_10)
      assert delta_10 == expected_10
    end

    test "handles receiving a delta message for a function_call" do
      delta_function = LangChain.Fixtures.raw_deltas_for_function_call()

      msg_1 = Enum.at(delta_function, 0)
      msg_2 = Enum.at(delta_function, 1)
      msg_3 = Enum.at(delta_function, 2)

      expected_1 = %MessageDelta{
        content: nil,
        index: 0,
        function_name: "hello_world",
        role: :assistant,
        arguments: "",
        status: :incomplete
      }

      [%MessageDelta{} = delta_1] = ChatOpenAI.do_process_response(msg_1)
      assert delta_1 == expected_1

      expected_2 = %MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: "{}",
        status: :incomplete
      }

      [%MessageDelta{} = delta_2] = ChatOpenAI.do_process_response(msg_2)
      assert delta_2 == expected_2

      expected_3 = %MessageDelta{
        content: nil,
        index: 0,
        function_name: nil,
        role: :unknown,
        arguments: nil,
        status: :complete
      }

      # it should not trim the arguments text
      [%MessageDelta{} = delta_3] = ChatOpenAI.do_process_response(msg_3)
      assert delta_3 == expected_3
    end

    # test "handles receiving error message from server"

    test "handles json parse error from server" do
      {:error, "Received invalid JSON: " <> _} =
        Jason.decode("invalid json")
        |> ChatOpenAI.do_process_response()
    end

    test "handles unexpected response" do
      {:error, "Unexpected response"} =
        "unexpected"
        |> ChatOpenAI.do_process_response()
    end

    test "return multiple responses when given multiple choices" do
      # received multiple responses because multiples were requested.
      response = %{
        "choices" => [
          %{
            "message" => %{"role" => "assistant", "content" => "Greetings!"},
            "finish_reason" => "stop",
            "index" => 0
          },
          %{
            "message" => %{"role" => "assistant", "content" => "Howdy!"},
            "finish_reason" => "stop",
            "index" => 1
          }
        ]
      }

      [msg1, msg2] = ChatOpenAI.do_process_response(response)
      assert %Message{role: :assistant, index: 0} = msg1
      assert %Message{role: :assistant, index: 1} = msg2
      assert msg1.content == "Greetings!"
      assert msg2.content == "Howdy!"
    end
  end

  describe "streaming examples" do
    @tag live_call: true, live_open_ai: true
    test "supports streaming response calling function with args" do
      callback = fn data ->
        # IO.inspect(data, label: "DATA")
        send(self(), {:streamed_fn, data})
      end

      {:ok, chat} = ChatOpenAI.new(%{seed: 0, stream: true})

      {:ok, message} =
        Message.new_user("Answer the following math question: What is 100 + 300 - 200?")

      _response =
        ChatOpenAI.do_api_request(chat, [message], [LangChain.Tools.Calculator.new!()], callback)

      # IO.inspect(response, label: "OPEN AI POST RESPONSE")

      assert_receive {:streamed_fn, received_data}, 300
      assert %MessageDelta{} = received_data
      assert received_data.role == :assistant
      assert received_data.index == 0
    end

    @tag live_call: true, live_open_ai: true
    test "STREAMING handles receiving an error when no messages sent" do
      chat = ChatOpenAI.new!(%{seed: 0, stream: true})

      {:error, reason} = ChatOpenAI.call(chat, [], [], nil)

      assert reason == "[] is too short - 'messages'"
    end

    @tag live_call: true, live_open_ai: true
    test "STREAMING handles receiving a timeout error" do
      callback = fn data ->
        send(self(), {:streamed_fn, data})
      end

      {:ok, chat} = ChatOpenAI.new(%{seed: 0, stream: true, receive_timeout: 50})

      {:error, reason} =
        ChatOpenAI.call(chat, [Message.new_user!("Why is the sky blue?")], [], callback)

      assert reason == "Request timed out"
    end
  end

  def setup_expected_json(_) do
    json_1 = %{
      "choices" => [
        %{
          "delta" => %{
            "content" => nil,
            "function_call" => %{"arguments" => "", "name" => "calculator"},
            "role" => "assistant"
          },
          "finish_reason" => nil,
          "index" => 0
        }
      ],
      "created" => 1_689_801_995,
      "id" => "chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS",
      "model" => "gpt-4-0613",
      "object" => "chat.completion.chunk"
    }

    json_2 = %{
      "choices" => [
        %{
          "delta" => %{"function_call" => %{"arguments" => "{\n"}},
          "finish_reason" => nil,
          "index" => 0
        }
      ],
      "created" => 1_689_801_995,
      "id" => "chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS",
      "model" => "gpt-4-0613",
      "object" => "chat.completion.chunk"
    }

    %{json_1: json_1, json_2: json_2}
  end

  describe "decode_stream/1" do
    setup :setup_expected_json

    test "correctly handles fully formed chat completion chunks", %{
      json_1: json_1,
      json_2: json_2
    } do
      data =
        "data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n
         data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"function_call\":{\"arguments\":\"{\\n\"}},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = ChatOpenAI.decode_stream({data, ""})

      # nothing incomplete. Parsed 2 objects.
      assert incomplete == ""
      assert parsed == [json_1, json_2]
    end

    test "correctly parses when data split over received messages", %{json_1: json_1} do
      # split the data over multiple messages
      data =
        "data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comple
         data: tion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = ChatOpenAI.decode_stream({data, ""})

      # nothing incomplete. Parsed 1 object.
      assert incomplete == ""
      assert parsed == [json_1]
    end

    test "correctly parses when data split over decode calls", %{json_1: json_1} do
      buffered = "{\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comple"

      # incomplete message chunk processed in next call
      data =
        "data: tion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n"

      {parsed, incomplete} = ChatOpenAI.decode_stream({data, buffered})

      # nothing incomplete. Parsed 1 object.
      assert incomplete == ""
      assert parsed == [json_1]
    end

    test "correctly parses when data previously buffered and responses split and has leftovers",
         %{json_1: json_1, json_2: json_2} do
      buffered = "{\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comple"

      # incomplete message chunk processed in next call
      data =
        "data: tion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n
         data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"crea
         data: ted\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"function_call\":{\"argu
         data: ments\":\"{\\n\"}},\"finish_reason\":null}]}\n\n
         data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comp"

      {parsed, incomplete} = ChatOpenAI.decode_stream({data, buffered})

      # nothing incomplete. Parsed 1 object.
      assert incomplete ==
               "{\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.comp"

      assert parsed == [json_1, json_2]
    end
  end
end
